"""
middleware.py
-------------
운영 관점의 안정성을 담당하는 미들웨어 레이어.

1. InputValidator  — 사용자 입력 길이/금지어 검증
2. AgentLogger     — 입출력 및 Tool 호출 이력을 파일 + 콘솔에 기록
3. ErrorHandler    — 예외를 잡아 상태에 기록하고 안전하게 복구
"""

import os
import re
import logging
import functools
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOG_PATH = os.getenv("LOG_PATH", "./output/agent.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ClinicalAgent")


# ── 1. 입력 검증 ───────────────────────────────────────────────────────────────
class InputValidationError(ValueError):
    pass


BLOCKED_PATTERNS = [
    r"ignore previous",
    r"ignore all instructions",
    r"프롬프트 무시",
    r"system prompt",
]

MAX_INPUT_LENGTH = 2000


def validate_input(user_input: str) -> str:
    """
    사용자 입력에 대해 두 가지를 검증한다.
    - 길이 초과 여부
    - 프롬프트 인젝션 패턴 포함 여부
    통과하면 strip 된 입력을 반환, 실패하면 InputValidationError 를 던진다.
    """
    text = user_input.strip()

    if len(text) == 0:
        raise InputValidationError("입력이 비어있습니다.")

    if len(text) > MAX_INPUT_LENGTH:
        raise InputValidationError(
            f"입력이 너무 깁니다. (최대 {MAX_INPUT_LENGTH}자, 현재 {len(text)}자)"
        )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise InputValidationError("허용되지 않는 입력 패턴이 감지되었습니다.")

    return text


# ── 2. 로거 ────────────────────────────────────────────────────────────────────
def log_user_input(user_input: str):
    logger.info(f"[USER INPUT] {user_input}")


def log_intent(intent: str):
    logger.info(f"[INTENT DETECTED] {intent}")


def log_tool_call(tool_name: str, tool_input: str):
    logger.info(f"[TOOL CALL] {tool_name} | input: {tool_input}")


def log_tool_result(tool_name: str, result_snippet: str):
    snippet = result_snippet[:200].replace("\n", " ")
    logger.info(f"[TOOL RESULT] {tool_name} | {snippet}...")


def log_final_report(patient_id: int | None):
    logger.info(f"[FINAL REPORT GENERATED] patient_id={patient_id}")


def log_error(context: str, error: Exception):
    logger.error(f"[ERROR] {context} | {type(error).__name__}: {error}")
    logger.debug(traceback.format_exc())


# ── 3. 에러 핸들러 데코레이터 ─────────────────────────────────────────────────
def safe_node(node_name: str):
    """
    LangGraph 노드 함수에 씌우는 데코레이터.
    예외 발생 시 상태의 error 필드에 메시지를 기록하고 그래프 실행을 중단하지 않는다.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            try:
                return func(state, *args, **kwargs)
            except Exception as e:
                log_error(node_name, e)
                return {**state.model_dump(), "error": f"[{node_name}] {type(e).__name__}: {e}"}
        return wrapper
    return decorator
