"""
graph.py
--------
LangGraph StateGraph 로 전체 Agent 실행 흐름을 설계한다.

변경사항 요약:
  1. [RAG 쿼리 정제]   rag_node — 사용자 발화를 그대로 임베딩하지 않고
                        LLM 으로 영문 의학 키워드만 추출 후 검색.
  2. [멀티턴 메모리]   intent_node — 이전 대화 이력을 system prompt 에 주입.
                        report_node — 완료 후 ConversationTurn 을 history 에 append.
  3. [대명사 참조]     intent_node — "이 환자", "방금 그 환자" 등
                        last_patient_id 로 자동 해소.
  4. [RAG 관련성 필터] rag_node — 검색 결과가 쿼리와 무관한 경우
                        "관련 노트 없음" 메시지 반환, LLM 환각 방지.
  5. [리포트 분기]     report_node — patient_id 有/無 에 따라 출력 템플릿 분리.
  6. [retry loop]      rag_node 결과 없을 시 → retry_node 에서 쿼리 재작성
                        → rag_node 재시도 (최대 MAX_RAG_RETRY 회)

노드 구성:
  input_node      → 입력 검증 + 환자 ID 파싱
  intent_node     → LLM 이 사용자 의도를 predict / explain / both / unknown 으로 분류
  gnn_node        → Tool A 호출 (GNN 질병 예측)
  rag_node        → Tool B 호출 (RAG 임상 노트 검색) + 쿼리 정제
  retry_node      → RAG 실패 시 쿼리 재작성 후 재시도 (loop)
  report_node     → GPT-4o-mini 로 최종 임상 리포트 합성
  error_node      → 에러 메시지를 사용자 친화적으로 반환

조건부 분기 (conditional edges):
  input_node  이후 → intent_node / error_node
  intent_node 이후 → gnn_node / rag_node / 둘 다 / error_node
  gnn_node    이후 → rag_node(both) / report_node
  rag_node    이후 → report_node(성공) / retry_node(실패) / error_node
  retry_node  이후 → rag_node(재시도) / report_node(한계 초과)
"""

import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from agent.state import AgentState, ConversationTurn
from tools.tools import gnn_predict_tool, rag_search_tool
from middleware.middleware import (
    validate_input, log_user_input, log_intent,
    log_tool_call, log_tool_result, log_final_report,
    log_error, safe_node, InputValidationError,
)

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────
MAX_RAG_RETRY = 2   # RAG 재시도 최대 횟수

# ── LLM 초기화 ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# ── OutputParser 스키마 ────────────────────────────────────────────────────────
class IntentOutput(BaseModel):
    """의도 분류 결과 구조화 출력."""
    intent: str = Field(description="predict | explain | both | unknown 중 하나")
    patient_id: int | None = Field(default=None, description="언급된 환자 ID (없으면 null)")
    reasoning: str = Field(description="분류 근거 한 줄 요약")


intent_parser = PydanticOutputParser(pydantic_object=IntentOutput)


# ── 헬퍼: 대화 이력 → 프롬프트 문자열 ────────────────────────────────────────
def format_history_for_prompt(history: list[ConversationTurn], max_turns: int = 5) -> str:
    if not history:
        return "없음"
    recent = history[-max_turns:]
    lines = []
    for t in recent:
        pid_info     = f"환자 {t.patient_id}" if t.patient_id else "환자 미지정"
        disease_info = f"→ 예측 1위: {t.top1_disease}" if t.top1_disease else ""
        lines.append(
            f"[턴 {t.turn_index + 1}] 사용자: '{t.user_input}' / 의도: {t.intent} / "
            f"{pid_info} {disease_info}"
        )
    return "\n".join(lines)


# ── 노드 1: 입력 검증 ─────────────────────────────────────────────────────────
@safe_node("input_node")
def input_node(state: AgentState) -> dict:
    last_msg  = state.messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    try:
        clean_text = validate_input(user_text)
        log_user_input(clean_text)
        return {"messages": state.messages, "error": None}
    except InputValidationError as e:
        log_error("input_node", e)
        return {"error": str(e)}


# ── 노드 2: 의도 분류 ─────────────────────────────────────────────────────────
@safe_node("intent_node")
def intent_node(state: AgentState) -> dict:
    last_msg  = state.messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    history_text      = format_history_for_prompt(state.conversation_history)
    format_instructions = intent_parser.get_format_instructions()

    system_prompt = f"""당신은 임상 AI 어시스턴트의 라우터입니다.
사용자 메시지를 분석하여 의도를 아래 중 하나로 분류하세요.

- predict  : 환자 ID 를 주고 질병 예측을 요청하는 경우
- explain  : 특정 질병/코드에 대한 설명이나 임상 노트 검색을 요청하는 경우.
             "이 환자 ~이랑 관련 있어?", "이 환자 ~이 뭐야?", "~에 대해 설명해줘" 등
             질병 키워드가 포함된 질문은 환자 ID 유무와 관계없이 explain 으로 분류하세요.
- both     : 예측 + 설명을 모두 요청하거나, 예측 후 해당 질병 설명도 원하는 경우
- unknown  : 인사, 잡담 등 임상과 전혀 무관한 경우에만 사용하세요.
             임상 관련 질문이라면 unknown 을 사용하지 마세요.

【이전 대화 이력】
{history_text}

【대명사 참조 규칙】
- "이 환자", "방금 그 환자", "같은 환자", "아까 그 환자" 등의 표현이 있으면
  이전 이력의 마지막 patient_id({state.last_patient_id})를 patient_id 로 사용하세요.
- patient_id 는 대화 이력에서 참조하며, 없으면 null 로 반환하세요.

【분류 예시】
- "환자 188번 예측해줘"                → predict,  patient_id: 188
- "폐렴 임상노트 검색해줘"             → explain,  patient_id: null
- "이 환자 폐렴이랑 관련 있어?"        → explain,  patient_id: {state.last_patient_id}
- "환자 188번 예측하고 설명도 해줘"    → both,     patient_id: 188
- "안녕하세요"                         → unknown,  patient_id: null

{format_instructions}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])

    parsed: IntentOutput = intent_parser.parse(response.content)
    log_intent(parsed.intent)

    resolved_patient_id = parsed.patient_id
    if resolved_patient_id is None and state.last_patient_id is not None:
        reference_keywords = ["이 환자", "그 환자", "방금", "다시", "같은 환자", "아까"]
        if any(kw in user_text for kw in reference_keywords):
            resolved_patient_id = state.last_patient_id

    return {
        "intent": parsed.intent,
        "patient_id": resolved_patient_id,
        "gnn_result": None,
        "rag_result": None,
        "rag_query_used": None,
        "rag_retry_count": 0,   # ← 재시도 카운터 리셋
        "error": None,
    }


# ── 노드 3: GNN 질병 예측 (Tool A) ───────────────────────────────────────────
@safe_node("gnn_node")
def gnn_node(state: AgentState) -> dict:
    if state.patient_id is None:
        return {"error": "환자 ID 를 찾을 수 없습니다. 예: '환자 10006 번 예측해줘'"}

    log_tool_call("gnn_predict_tool", str(state.patient_id))
    result_str  = gnn_predict_tool.invoke({"patient_id": state.patient_id})
    log_tool_result("gnn_predict_tool", result_str)

    result_dict = json.loads(result_str)
    return {"gnn_result": result_dict}


# ── 노드 4: RAG 임상 노트 검색 (Tool B) + 쿼리 정제 ─────────────────────────
@safe_node("rag_node")
def rag_node(state: AgentState) -> dict:
    """
    검색 쿼리 결정 우선순위:
      1. GNN 결과 있음  → top1 질병명 + ICD9 코드 (영문, 깨끗함)
      2. retry 상태     → retry_node 가 재작성한 rag_query_used 사용
      3. 그 외          → LLM 으로 사용자 발화에서 영문 의학 키워드 추출
    """
    if state.gnn_result and state.gnn_result.get("top5_predictions"):
        top1  = state.gnn_result["top5_predictions"][0]
        query = f"{top1['disease_name']} {top1['icd9_code']}"

    elif state.rag_retry_count > 0 and state.rag_query_used:
        # retry_node 가 이미 새 쿼리를 rag_query_used 에 써 넣었음
        query = state.rag_query_used

    else:
        last_msg  = state.messages[-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        extraction_response = llm.invoke([
            SystemMessage(content="""You are a medical term extractor for a clinical NLP system.
Extract ONLY the core medical concept from the user's message.
Return ONLY the English medical term or disease name. No explanation, no Korean, no extra words.

Examples:
- "폐렴에 대해 임상노트 검색해줘" → "pneumonia"
- "심부전 임상노트 알려줘" → "congestive heart failure"
- "패혈증이 뭔지 설명해줘" → "sepsis"
- "당뇨병 관련 노트 보여줘" → "diabetes mellitus"
- "급성 신부전 케이스 찾아줘" → "acute kidney failure"
- "MRSA 감염 노트" → "MRSA infection"
"""),
            HumanMessage(content=user_text),
        ])
        query = extraction_response.content.strip()

    log_tool_call("rag_search_tool", query)
    result_str = rag_search_tool.invoke({"query": query})
    log_tool_result("rag_search_tool", result_str)

    if not result_str or result_str.strip() == "":
        result_str = f"[관련 임상 노트 없음] '{query}' 에 해당하는 임상 기록을 찾지 못했습니다."

    return {
        "rag_result": result_str,
        "rag_query_used": query,
    }


# ── 노드 5: RAG 재시도 — 쿼리 재작성 (loop) ─────────────────────────────────
@safe_node("retry_node")
def retry_node(state: AgentState) -> dict:
    """
    RAG 검색이 실패했을 때 호출된다.
    LLM 으로 이전 쿼리를 분석하여 더 넓은 의미의 쿼리로 재작성한 후
    rag_query_used 에 저장 → rag_node 가 이를 읽어 재시도.
    """
    old_query = state.rag_query_used or ""
    retry_num = state.rag_retry_count + 1

    rewrite_response = llm.invoke([
        SystemMessage(content="""You are a medical search query optimizer.
The previous query returned no results from a clinical notes database.
Rewrite it as a broader or alternative English medical term.
Return ONLY the new query. No explanation.

Examples:
- "diabetes mellitus type 2" → "diabetes"
- "acute myocardial infarction" → "heart attack myocardial"
- "MRSA infection" → "staphylococcal infection"
"""),
        HumanMessage(content=f"Previous query: {old_query}"),
    ])

    new_query = rewrite_response.content.strip()
    print(f"🔄 RAG 재시도 [{retry_num}/{MAX_RAG_RETRY}]: '{old_query}' → '{new_query}'")

    return {
        "rag_query_used": new_query,
        "rag_retry_count": retry_num,
        "rag_result": None,     # 이전 실패 결과 초기화
    }


# ── 노드 6: 최종 리포트 합성 ──────────────────────────────────────────────────
@safe_node("report_node")
def report_node(state: AgentState) -> dict:
    sections    = []
    has_patient = state.patient_id is not None
    top1_disease = None

    if state.gnn_result and "top5_predictions" in state.gnn_result:
        pid          = state.gnn_result.get("patient_id", "?")
        preds        = state.gnn_result["top5_predictions"]
        vital        = state.gnn_result.get("vital_summary", "바이탈 정보 없음")
        ground_truth = state.gnn_result.get("actual_ground_truth", [])
        top1_disease = preds[0]["disease_name"] if preds else None

        # ── 실제 진단 정답 포맷팅 ─────────────────────────────────────────────
        if ground_truth:
            gt_text = "\n".join([
                f"  • [{g['icd9_code']}] {g['disease_name']}"
                for g in ground_truth
            ])
        else:
            gt_text = "  정답 데이터 없음"

        # ── GNN 예측 포맷팅 (정답 일치 여부 표시) ────────────────────────────
        pred_lines = []
        for p in preds:
            hit   = "✅ 정답 일치" if p.get("matched") else "  "
            pred_lines.append(
                f"  {p['rank']}위: [{p['icd9_code']}] {p['disease_name']} "
                f"(유사도 {p['cosine_similarity']:.4f}) {hit}"
            )
        pred_text = "\n".join(pred_lines)

        sections.append(
            f"【실제 진단 정답 — 환자 {pid}】\n{gt_text}\n\n"
            f"【GNN 유사도 기반 예측 질병 Top-5】\n{pred_text}\n\n"
            f"【24시간 바이탈 요약】\n{vital}"
        )

    if state.rag_result:
        retry_info = f" (재시도 {state.rag_retry_count}회)" if state.rag_retry_count > 0 else ""
        query_info = f"(검색어: {state.rag_query_used}{retry_info})" if state.rag_query_used else ""
        sections.append(f"【유사 임상 노트 (RAG 검색) {query_info}】\n{state.rag_result}")

    combined_context = "\n\n" + ("=" * 50) + "\n\n".join(sections)

    if has_patient:
        system_prompt = """당신은 ICU 임상 추론 AI 입니다.
아래 제공된 데이터를 바탕으로 간결하고 전문적인 임상 소견 요약을 작성하세요.

데이터 구성:
1. 실제 진단 정답: 해당 환자가 실제로 진단받은 질병 목록
2. GNN 유사도 기반 예측 질병 Top-5: 임베딩 유사도로 예측한 질병 (✅ 표시 = 정답과 일치)
3. 24시간 바이탈 요약

작성 지침:
- 실제 진단 정답 질병들을 먼저 간략히 설명하세요.
- GNN 예측 결과 중 정답과 일치한 항목(✅)은 "예측 성공"으로, 불일치 항목은 "추가 발현 가능 질병"으로 구분하여 설명하세요.
- 바이탈 패턴과 진단 질병 간의 임상적 연관성을 설명하세요.
- 없는 정보(나이, 성별 등)를 추측하여 작성하지 마세요.
- GNN 예측 유사도가 전반적으로 0.5 미만이면 "모델 신뢰도가 낮으므로 추가 검사가 필요합니다"라고 명시하세요.
- 임상적으로 주의해야 할 사항을 마지막에 간략히 언급하세요."""
    else:
        system_prompt = """당신은 임상 의학 정보 AI 입니다.
아래 유사 임상 노트를 바탕으로 해당 질병에 대한 임상 정보를 요약하세요.
- 검색된 임상 노트에서 발견되는 주요 증상, 치료, 검사 패턴을 정리하세요.
- 실제 임상 노트 내용을 근거로만 설명하고, 없는 정보를 창작하지 마세요.
- '[관련 임상 노트 없음]' 이 포함된 경우 솔직하게 "관련 노트를 찾지 못했습니다" 라고 안내하세요.
- 마지막에 해당 질병에서 임상적으로 주의할 점을 1~2문장으로 정리하세요."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=combined_context),
    ])

    final_report = response.content
    log_final_report(state.patient_id)

    last_msg  = state.messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    current_turn = ConversationTurn(
        turn_index=state.turn_count,
        user_input=user_text,
        intent=state.intent,
        patient_id=state.patient_id,
        top1_disease=top1_disease,
        rag_query=state.rag_query_used,
        final_report=final_report[:300],
    )

    return {
        "final_report": final_report,
        "messages": state.messages + [AIMessage(content=final_report)],
        "conversation_history": [current_turn],
        "turn_count": state.turn_count + 1,
        "last_patient_id": state.patient_id if state.patient_id else state.last_patient_id,
        "last_intent": state.intent,
    }


# ── 노드 7: 에러 응답 ─────────────────────────────────────────────────────────
def error_node(state: AgentState) -> dict:
    error_msg = state.error or "알 수 없는 오류가 발생했습니다."
    reply = f"죄송합니다. 처리 중 문제가 발생했습니다.\n\n오류: {error_msg}"
    return {
        "final_report": reply,
        "messages": state.messages + [AIMessage(content=reply)],
    }


# ── 조건부 분기 함수 ──────────────────────────────────────────────────────────
def route_after_input(state: AgentState) -> str:
    if state.error:
        return "error_node"
    return "intent_node"


def route_after_intent(state: AgentState) -> str:
    if state.error:
        return "error_node"
    intent = state.intent
    if intent == "predict":
        return "gnn_node"
    elif intent == "explain":
        return "rag_node"
    elif intent == "both":
        return "gnn_node"
    else:
        return "error_node"


def route_after_gnn(state: AgentState) -> str:
    if state.error:
        return "error_node"
    if state.intent == "both":
        return "rag_node"
    return "report_node"


def route_after_rag(state: AgentState) -> str:
    """
    RAG 성공 여부에 따라 분기:
      - 결과 있음          → report_node
      - 결과 없음 + 재시도 가능 → retry_node (loop)
      - 결과 없음 + 한계 초과  → report_node (없음 안내)
    """
    if state.error:
        return "error_node"

    rag_failed = (
        not state.rag_result
        or "관련 임상 노트를 찾지 못했습니다" in state.rag_result
        or "관련 노트를 찾지 못했습니다" in state.rag_result
    )

    if rag_failed and state.rag_retry_count < MAX_RAG_RETRY:
        return "retry_node"

    return "report_node"


def route_after_retry(state: AgentState) -> str:
    """retry_node 완료 후 항상 rag_node 로 돌아감 (loop)."""
    if state.error:
        return "error_node"
    return "rag_node"


# ── 그래프 조립 ───────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("input_node",  input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("gnn_node",    gnn_node)
    graph.add_node("rag_node",    rag_node)
    graph.add_node("retry_node",  retry_node)
    graph.add_node("report_node", report_node)
    graph.add_node("error_node",  error_node)

    graph.set_entry_point("input_node")

    graph.add_conditional_edges(
        "input_node",
        route_after_input,
        {"intent_node": "intent_node", "error_node": "error_node"},
    )
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {"gnn_node": "gnn_node", "rag_node": "rag_node", "error_node": "error_node"},
    )
    graph.add_conditional_edges(
        "gnn_node",
        route_after_gnn,
        {"rag_node": "rag_node", "report_node": "report_node", "error_node": "error_node"},
    )
    # ✅ rag_node 이후 조건부 분기 (loop 포함)
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {"report_node": "report_node", "retry_node": "retry_node", "error_node": "error_node"},
    )
    # ✅ retry_node → rag_node (loop)
    graph.add_conditional_edges(
        "retry_node",
        route_after_retry,
        {"rag_node": "rag_node", "error_node": "error_node"},
    )

    graph.add_edge("report_node", END)
    graph.add_edge("error_node",  END)

    return graph.compile()


clinical_agent = build_graph()
