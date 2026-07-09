"""
main.py
-------
ClinicalGNN-LLM Agent 실행 진입점.

멀티턴 대화 루프를 실행하고, 대화 이력(메모리)을 세션 동안 유지한다.

실행:
    python main.py
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.graph import clinical_agent
from agent.state import AgentState
from middleware.middleware import logger

load_dotenv()


def print_banner():
    print("\n" + "=" * 60)
    print("  ClinicalGNN-LLM Agent  |  MIMIC-III 임상 추론 시스템")
    print("=" * 60)
    print("사용 예시:")
    print("  '환자 10006번 질병 예측해줘'              → GNN 예측")
    print("  '폐렴에 대해 임상 노트 검색해줘'          → RAG 검색")
    print("  '환자 10006번 예측하고 질병 설명도 해줘'  → 예측 + 검색")
    print("  'quit' 또는 'exit' 입력 시 종료")
    print("=" * 60 + "\n")


def run():
    print_banner()

    # ✅ 핵심 수정: AgentState를 세션 시작 시 1회만 생성
    # 매 턴마다 새로 만들지 않고 누적 상태(conversation_history,
    # last_patient_id, turn_count 등)를 그대로 유지한다.
    session_state = AgentState()

    while True:
        try:
            user_input = input("🧑 사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료"):
            print("세션을 종료합니다.")
            break

        # ✅ 현재 메시지만 추가한 새 state 생성
        # 누적 필드(conversation_history, last_patient_id 등)는
        # session_state 에서 그대로 가져오고, messages 에만 새 입력을 넣는다.
        current_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            # ── 누적 필드 이어받기 ──────────────────────────────────────────
            conversation_history=session_state.conversation_history,
            turn_count=session_state.turn_count,
            last_patient_id=session_state.last_patient_id,
            last_intent=session_state.last_intent,
        )

        try:
            # 그래프 실행
            final_state = clinical_agent.invoke(current_state)

            # 최종 리포트 출력
            report = final_state.get("final_report") or "응답을 생성하지 못했습니다."
            print(f"\n🤖 Agent:\n{report}\n")
            print("-" * 60)

            # ✅ 누적 상태 session_state 에 반영
            session_state = AgentState(
                messages=final_state.get("messages", []),
                conversation_history=final_state.get("conversation_history", session_state.conversation_history),
                turn_count=final_state.get("turn_count", session_state.turn_count),
                last_patient_id=final_state.get("last_patient_id", session_state.last_patient_id),
                last_intent=final_state.get("last_intent", session_state.last_intent),
            )

            # 에러 발생 시 로그
            if final_state.get("error"):
                logger.warning(f"세션 중 에러 발생: {final_state['error']}")

        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}\n")
            logger.error(f"main loop error: {e}")


if __name__ == "__main__":
    run()
