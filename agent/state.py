"""
LangGraph StateGraph 에서 노드 간에 공유되는 상태(State) 스키마 정의.
모든 노드는 이 상태를 읽고 업데이트한다.

변경사항:
- ConversationTurn : 한 턴의 대화 기록 구조체 추가
- conversation_history : Annotated + _append_turns reducer 로 누적 append 보장
- turn_count : 현재까지 진행된 턴 수
- last_patient_id / last_intent : 멀티턴 대명사 참조용
"""

from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class ConversationTurn(BaseModel):
    """
    한 턴(사용자 질의 1회 + Agent 응답 1회)의 기록.
    멀티턴 대화에서 이전 문맥을 참조할 때 사용한다.
    """
    turn_index: int = Field(description="턴 번호 (0-based)")
    user_input: str = Field(description="사용자 발화 원문")
    intent: str = Field(description="분류된 의도")
    patient_id: int | None = Field(default=None, description="해당 턴의 환자 ID")
    top1_disease: str | None = Field(default=None, description="GNN 예측 1위 질병명")
    rag_query: str | None = Field(default=None, description="실제 사용된 RAG 검색어")
    final_report: str | None = Field(default=None, description="최종 리포트 요약")


# ✅ 변경 1: reducer 함수 추가
# LangGraph 는 list 필드를 기본으로 덮어쓰기(override) 한다.
# 이 함수를 Annotated 에 달면 "새 항목만 넘겨도 누적 append" 가 보장된다.
def _append_turns(existing: list, new: list) -> list:
    return existing + new


class AgentState(BaseModel):
    """
    멀티턴 대화 전반에 걸쳐 유지되는 통합 상태 객체.

    [현재 턴 상태] — 매 턴마다 리셋
    - messages          : LangGraph add_messages reducer (원본 메시지 누적)
    - patient_id        : 현재 분석 대상 환자 ID
    - intent            : 라우터가 판별한 사용자 의도
    - gnn_result        : Tool A (GNN 예측) 결과 dict
    - rag_result        : Tool B (RAG 검색) 결과 str
    - rag_query_used    : 실제 RAG에 넘긴 검색어 (디버깅/로깅용)
    - final_report      : 최종 합성 리포트 문자열
    - error             : 에러 메시지 (발생 시)

    [멀티턴 누적 상태] — 세션 동안 누적
    - conversation_history : 이전 턴 기록 리스트 (ConversationTurn)
    - turn_count           : 진행된 총 턴 수
    - last_patient_id      : 직전 턴에서 사용된 환자 ID
                             (예: "이 환자 다시 분석해줘" 같은 참조 질의 처리)
    - last_intent          : 직전 턴의 의도
    """

    # ── 현재 턴 상태 ──────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    patient_id: int | None = None
    intent: Literal["predict", "explain", "both", "unknown"] = "unknown"
    gnn_result: dict | None = None
    rag_result: str | None = None
    rag_query_used: str | None = None
    final_report: str | None = None
    error: str | None = None

    # ── 멀티턴 누적 상태 ─────────────────────────────────────────────────────
    # ✅ 변경 2: list[ConversationTurn] → Annotated[..., _append_turns]
    # report_node 에서 [current_turn] 만 반환해도 기존 이력에 누적된다.
    conversation_history: Annotated[list[ConversationTurn], _append_turns] = Field(
        default_factory=list,
        description="세션 내 전체 대화 이력. report_node 완료 시 append.",
    )
    turn_count: int = Field(default=0, description="완료된 턴 수")
    last_patient_id: int | None = Field(
        default=None,
        description="직전 턴의 환자 ID. '이 환자' 등 대명사 참조 시 활용.",
    )
    last_intent: str = Field(default="unknown", description="직전 턴의 의도")

    class Config:
        arbitrary_types_allowed = True
