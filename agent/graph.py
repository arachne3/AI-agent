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

노드 구성:
  input_node      → 입력 검증 + 환자 ID 파싱
  intent_node     → LLM 이 사용자 의도를 predict / explain / both / unknown 으로 분류
  gnn_node        → Tool A 호출 (GNN 질병 예측)
  rag_node        → Tool B 호출 (RAG 임상 노트 검색) + 쿼리 정제
  report_node     → GPT-4o-mini 로 최종 임상 리포트 합성
  error_node      → 에러 메시지를 사용자 친화적으로 반환

조건부 분기 (conditional edges):
  intent_node 이후 → intent 값에 따라 gnn_node / rag_node / 둘 다 / error_node 로 분기
  gnn_node 이후   → both 면 rag_node 로, 아니면 report_node 로
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
    """
    최근 max_turns 개의 ConversationTurn 을 LLM 프롬프트용 텍스트로 변환.
    오래된 이력은 잘라내어 토큰 낭비를 방지한다.
    """
    if not history:
        return "없음"

    recent = history[-max_turns:]
    lines = []
    for t in recent:
        pid_info = f"환자 {t.patient_id}" if t.patient_id else "환자 미지정"
        disease_info = f"→ 예측 1위: {t.top1_disease}" if t.top1_disease else ""
        lines.append(
            f"[턴 {t.turn_index + 1}] 사용자: '{t.user_input}' / 의도: {t.intent} / "
            f"{pid_info} {disease_info}"
        )
    return "\n".join(lines)


# ── 노드 1: 입력 검증 ─────────────────────────────────────────────────────────
@safe_node("input_node")
def input_node(state: AgentState) -> dict:
    last_msg = state.messages[-1]
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
    last_msg = state.messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # ── 멀티턴: 이전 대화 이력을 프롬프트에 주입 ─────────────────────────────
    history_text = format_history_for_prompt(state.conversation_history)
    format_instructions = intent_parser.get_format_instructions()

    system_prompt = f"""당신은 임상 AI 어시스턴트의 라우터입니다.
사용자 메시지를 분석하여 의도를 아래 중 하나로 분류하세요.

- predict  : 환자 ID 를 주고 질병 예측을 요청하는 경우
- explain  : 특정 질병/코드에 대한 설명이나 임상 노트 검색을 요청하는 경우
- both     : 예측 + 설명을 모두 요청하거나, 예측 후 해당 질병 설명도 원하는 경우
- unknown  : 위 어디에도 해당하지 않는 경우

【이전 대화 이력】
{history_text}

【대명사 참조 규칙】
- "이 환자", "방금 그 환자", "같은 환자" 등의 표현이 있으면
  이전 이력의 마지막 patient_id({state.last_patient_id})를 patient_id 로 사용하세요.
- 환자 ID 가 명시되지 않고 이전 이력도 없으면 patient_id 는 null 로 반환하세요.

{format_instructions}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])

    parsed: IntentOutput = intent_parser.parse(response.content)
    log_intent(parsed.intent)

    # 환자 ID 가 null 이고 이전 이력에서 참조 가능한 경우 자동 채움
    resolved_patient_id = parsed.patient_id
    if resolved_patient_id is None and state.last_patient_id is not None:
        # "이 환자", "다시", "같은" 등의 표현이 있을 때만 채움
        reference_keywords = ["이 환자", "그 환자", "방금", "다시", "같은 환자", "아까"]
        if any(kw in user_text for kw in reference_keywords):
            resolved_patient_id = state.last_patient_id

    return {
        "intent": parsed.intent,
        "patient_id": resolved_patient_id,
        # 현재 턴 상태 리셋 (이전 턴 잔재 제거)
        "gnn_result": None,
        "rag_result": None,
        "rag_query_used": None,
        "error": None,
    }


# ── 노드 3: GNN 질병 예측 (Tool A) ───────────────────────────────────────────
@safe_node("gnn_node")
def gnn_node(state: AgentState) -> dict:
    if state.patient_id is None:
        return {"error": "환자 ID 를 찾을 수 없습니다. 예: '환자 10006 번 예측해줘'"}

    log_tool_call("gnn_predict_tool", str(state.patient_id))
    result_str = gnn_predict_tool.invoke({"patient_id": state.patient_id})
    log_tool_result("gnn_predict_tool", result_str)

    result_dict = json.loads(result_str)
    return {"gnn_result": result_dict}


# ── 노드 4: RAG 임상 노트 검색 (Tool B) + 쿼리 정제 ─────────────────────────
@safe_node("rag_node")
def rag_node(state: AgentState) -> dict:
    """
    검색 쿼리 결정 우선순위:
      1. GNN 결과가 있으면 → top1 질병명 + ICD9 코드 (영문, 깨끗함)
      2. GNN 결과가 없으면 → LLM 으로 사용자 발화에서 영문 의학 키워드 추출
         (한국어 명령어 노이즈 제거 + 영어 정규화)
    """
    if state.gnn_result and state.gnn_result.get("top5_predictions"):
        top1 = state.gnn_result["top5_predictions"][0]
        query = f"{top1['disease_name']} {top1['icd9_code']}"
    else:
        # ── 쿼리 정제: 사용자 발화 → 영문 의학 키워드 ────────────────────────
        last_msg = state.messages[-1]
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

    # ── 관련성 필터: tools.py 에서 이미 score 필터링 완료
    # "관련 임상 노트를 찾지 못했습니다" 로 시작하면 무관한 결과로 처리
    # → report_node 의 system prompt 가 "관련 노트 없음" 케이스를 별도 안내
    if not result_str or result_str.strip() == "":
        result_str = f"[관련 임상 노트 없음] '{query}' 에 해당하는 임상 기록을 찾지 못했습니다."

    return {
        "rag_result": result_str,
        "rag_query_used": query,    # 실제 사용된 검색어 저장
    }


# ── 노드 5: 최종 리포트 합성 ──────────────────────────────────────────────────
@safe_node("report_node")
def report_node(state: AgentState) -> dict:
    sections = []
    has_patient = state.patient_id is not None

    # ── GNN 예측 결과 포맷팅 ──────────────────────────────────────────────────
    top1_disease = None
    if state.gnn_result and "top5_predictions" in state.gnn_result:
        pid = state.gnn_result.get("patient_id", "?")
        preds = state.gnn_result["top5_predictions"]
        vital = state.gnn_result.get("vital_summary", "바이탈 정보 없음")
        top1_disease = preds[0]["disease_name"] if preds else None

        pred_text = "\n".join([
            f"  {p['rank']}위: [{p['icd9_code']}] {p['disease_name']} (유사도 {p['cosine_similarity']:.4f})"
            for p in preds
        ])
        sections.append(
            f"【GNN 예측 결과 — 환자 {pid}】\n{pred_text}\n\n【24시간 바이탈 요약】\n{vital}"
        )

    # ── RAG 검색 결과 포맷팅 ──────────────────────────────────────────────────
    if state.rag_result:
        query_info = f"(검색어: {state.rag_query_used})" if state.rag_query_used else ""
        sections.append(f"【유사 임상 노트 (RAG 검색) {query_info}】\n{state.rag_result}")

    combined_context = "\n\n" + ("=" * 50) + "\n\n".join(sections)

    # ── 리포트 템플릿 분기: 환자 예측 vs 질병 개념 검색 ──────────────────────
    if has_patient:
        system_prompt = """당신은 ICU 임상 추론 AI 입니다.
아래 제공된 GNN 예측 결과와 유사 임상 노트를 바탕으로 간결하고 전문적인 임상 소견 요약을 작성하세요.
- 없는 정보(나이, 성별 등)를 추측하여 작성하지 마세요.
- 예측된 질병의 주요 바이탈 패턴과의 연관성을 설명하세요.
- 임상적으로 주의해야 할 사항을 마지막에 간략히 언급하세요.
- GNN 예측 유사도가 0.5 미만인 경우 "예측 신뢰도가 낮으므로 추가 검사가 필요합니다" 라고 명시하세요."""
    else:
        # 질병 개념 검색 전용 템플릿
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

    # ── 멀티턴: 이번 턴 기록을 history 에 append ──────────────────────────────
    last_msg = state.messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    current_turn = ConversationTurn(
        turn_index=state.turn_count,
        user_input=user_text,
        intent=state.intent,
        patient_id=state.patient_id,
        top1_disease=top1_disease,
        rag_query=state.rag_query_used,
        final_report=final_report[:300],    # 너무 길면 요약만 저장
    )

    return {
        "final_report": final_report,
        "messages": state.messages + [AIMessage(content=final_report)],
        # ── 멀티턴 누적 상태 업데이트 ────────────────────────────────────────
        "conversation_history": [current_turn],
        "turn_count": state.turn_count + 1,
        "last_patient_id": state.patient_id if state.patient_id else state.last_patient_id,
        "last_intent": state.intent,
    }


# ── 노드 6: 에러 응답 ─────────────────────────────────────────────────────────
def error_node(state: AgentState) -> dict:
    error_msg = state.error or "알 수 없는 오류가 발생했습니다."
    reply = f"죄송합니다. 처리 중 문제가 발생했습니다.\n\n오류: {error_msg}"
    return {
        "final_report": reply,
        "messages": state.messages + [AIMessage(content=reply)],
    }


# ── 조건부 분기 함수 ──────────────────────────────────────────────────────────
def route_after_input(state: AgentState) -> str:
    """입력 검증 실패 시 error_node, 성공 시 intent_node."""
    if state.error:
        return "error_node"
    return "intent_node"


def route_after_intent(state: AgentState) -> str:
    """의도에 따라 gnn_node / rag_node / error_node 로 분기."""
    if state.error:
        return "error_node"
    intent = state.intent
    if intent == "predict":
        return "gnn_node"
    elif intent == "explain":
        return "rag_node"
    elif intent == "both":
        return "gnn_node"        # gnn → rag → report 순으로
    else:
        return "error_node"


def route_after_gnn(state: AgentState) -> str:
    """GNN 완료 후: both 면 rag_node, predict 면 바로 report_node."""
    if state.error:
        return "error_node"
    if state.intent == "both":
        return "rag_node"
    return "report_node"


# ── 그래프 조립 ───────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("input_node",  input_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("gnn_node",    gnn_node)
    graph.add_node("rag_node",    rag_node)
    graph.add_node("report_node", report_node)
    graph.add_node("error_node",  error_node)

    # 시작점
    graph.set_entry_point("input_node")

    # 조건부 엣지 1: input → intent or error
    graph.add_conditional_edges(
        "input_node",
        route_after_input,
        {"intent_node": "intent_node", "error_node": "error_node"},
    )

    # 조건부 엣지 2: intent → gnn / rag / error
    graph.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "gnn_node":   "gnn_node",
            "rag_node":   "rag_node",
            "error_node": "error_node",
        },
    )

    # 조건부 엣지 3: gnn → rag(both) or report
    graph.add_conditional_edges(
        "gnn_node",
        route_after_gnn,
        {"rag_node": "rag_node", "report_node": "report_node", "error_node": "error_node"},
    )

    # 일반 엣지
    graph.add_edge("rag_node",    "report_node")
    graph.add_edge("report_node", END)
    graph.add_edge("error_node",  END)

    return graph.compile()


# 외부에서 바로 사용할 수 있도록 컴파일된 그래프 노출
clinical_agent = build_graph()
