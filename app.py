"""
app.py
------
Flask 웹 서버 - ClinicalGNN-LLM Agent 웹 인터페이스

실행:
    pip install flask
    python app.py

접속:
    http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string, session
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os, uuid, json

from agent.graph import clinical_agent
from agent.state import AgentState

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)

# 세션별 AgentState 저장소 (메모리)
_sessions: dict[str, AgentState] = {}


def get_session_state(sid: str) -> AgentState:
    if sid not in _sessions:
        _sessions[sid] = AgentState()
    return _sessions[sid]


# ── HTML 템플릿 ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClinicalGNN-LLM Agent</title>
<style>
  /* ── 토큰 시스템 ─────────────────────────────────────────────── */
  :root {
    --bg-base:     #0b0f1a;   /* 깊은 네이비 블랙 */
    --bg-panel:    #111827;   /* 패널 배경 */
    --bg-input:    #1a2236;   /* 입력 영역 */
    --bg-bubble-u: #1e2d45;   /* 사용자 말풍선 */
    --bg-bubble-a: #131c2e;   /* 에이전트 말풍선 */
    --accent:      #38bdf8;   /* 하늘색 — 모니터 신호 느낌 */
    --accent-dim:  #0ea5e9;
    --warn:        #f59e0b;   /* 경고 amber */
    --ok:          #34d399;   /* 정상 green */
    --err:         #f87171;   /* 에러 red */
    --text-pri:    #e2e8f0;
    --text-sec:    #94a3b8;
    --text-dim:    #475569;
    --border:      #1e3a5f;
    --radius:      10px;
    --font-mono:   'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    --font-body:   'Inter', 'Pretendard', system-ui, sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg-base);
    color: var(--text-pri);
    font-family: var(--font-body);
    font-size: 14px;
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── 헤더 ─────────────────────────────────────────────────────── */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 24px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .logo-pulse {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--ok); }
    50%       { opacity: 0.4; box-shadow: 0 0 2px var(--ok); }
  }

  header h1 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-pri);
  }

  header .sub {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    margin-left: auto;
  }

  .badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--text-sec);
    letter-spacing: 0.05em;
  }

  /* ── 채팅 영역 ─────────────────────────────────────────────────── */
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    scroll-behavior: smooth;
  }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg-row {
    display: flex;
    gap: 12px;
    padding: 6px 24px;
    max-width: 860px;
    margin: 0 auto;
    width: 100%;
  }

  /* 아바타 */
  .avatar {
    width: 30px; height: 30px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .avatar-user  { background: var(--bg-bubble-u); border: 1px solid var(--border); }
  .avatar-agent { background: #0c1f35; border: 1px solid var(--accent); color: var(--accent); }

  /* 말풍선 */
  .bubble {
    flex: 1;
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.7;
    font-size: 13.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble-user {
    background: var(--bg-bubble-u);
    border: 1px solid var(--border);
    color: var(--text-pri);
  }
  .bubble-agent {
    background: var(--bg-bubble-a);
    border: 1px solid #1a3050;
    color: var(--text-pri);
  }

  /* 메타 레이블 */
  .msg-meta {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    margin-bottom: 4px;
    letter-spacing: 0.04em;
  }
  .msg-meta .role { color: var(--accent); }

  /* ── 시스템 메시지 ────────────────────────────────────────────── */
  .sys-msg {
    max-width: 860px;
    margin: 8px auto;
    padding: 6px 24px;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sys-msg::before {
    content: '';
    display: block;
    width: 24px; height: 1px;
    background: var(--border);
  }
  .sys-msg::after {
    content: '';
    display: block;
    flex: 1; height: 1px;
    background: var(--border);
  }

  /* ── 로딩 도트 ───────────────────────────────────────────────── */
  .typing {
    display: flex; gap: 5px; align-items: center;
    padding: 14px 16px;
    background: var(--bg-bubble-a);
    border: 1px solid #1a3050;
    border-radius: var(--radius);
  }
  .typing span {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: dot 1.2s ease-in-out infinite;
    opacity: 0.3;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes dot {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.2; }
    40%           { transform: scale(1.1); opacity: 1; }
  }

  /* ── 예시 칩 ────────────────────────────────────────────────── */
  #examples {
    max-width: 860px;
    margin: 32px auto 0;
    padding: 0 24px;
  }
  #examples p {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip {
    font-size: 12px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    color: var(--text-sec);
    background: var(--bg-panel);
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    font-family: var(--font-mono);
  }
  .chip:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── 입력 바 ────────────────────────────────────────────────── */
  #input-bar {
    flex-shrink: 0;
    background: var(--bg-panel);
    border-top: 1px solid var(--border);
    padding: 16px 24px;
  }
  #input-wrap {
    max-width: 860px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  #user-input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text-pri);
    font-family: var(--font-body);
    font-size: 14px;
    padding: 11px 16px;
    resize: none;
    outline: none;
    transition: border-color 0.15s;
    min-height: 44px;
    max-height: 140px;
    line-height: 1.5;
  }
  #user-input::placeholder { color: var(--text-dim); }
  #user-input:focus { border-color: var(--accent); }

  #send-btn {
    width: 44px; height: 44px;
    background: var(--accent);
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
    color: #0b0f1a;
  }
  #send-btn:hover:not(:disabled) { background: var(--accent-dim); }
  #send-btn:active:not(:disabled) { transform: scale(0.94); }
  #send-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  #send-btn svg { width: 18px; height: 18px; }

  /* ── 에러 토스트 ─────────────────────────────────────────────── */
  #toast {
    position: fixed;
    bottom: 80px; left: 50%;
    transform: translateX(-50%) translateY(10px);
    background: #3b0f0f;
    border: 1px solid var(--err);
    color: var(--err);
    font-size: 12px;
    font-family: var(--font-mono);
    padding: 10px 20px;
    border-radius: 8px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s, transform 0.2s;
    z-index: 99;
  }
  #toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  /* ── 반응형 ────────────────────────────────────────────────── */
  @media (max-width: 600px) {
    header { padding: 12px 16px; }
    #input-bar { padding: 12px 16px; }
    .msg-row, .sys-msg { padding-left: 16px; padding-right: 16px; }
    #examples { padding: 0 16px; }
  }
</style>
</head>
<body>

<!-- 헤더 -->
<header>
  <div class="logo-pulse"></div>
  <h1>ClinicalGNN-LLM Agent</h1>
  <span class="badge">MIMIC-III</span>
  <span class="sub" id="turn-counter">TURN 0</span>
</header>

<!-- 채팅 -->
<div id="chat">
  <div id="examples">
    <p>예시 질의</p>
    <div class="chip-row">
      <div class="chip" onclick="fillInput('환자 1197번 예측해줘')">환자 1197번 예측</div>
      <div class="chip" onclick="fillInput('폐렴에 대해 임상노트 검색해줘')">폐렴 임상노트 검색</div>
      <div class="chip" onclick="fillInput('환자 10006번 예측하고 질병 설명도 해줘')">예측 + 설명</div>
      <div class="chip" onclick="fillInput('이 환자 심부전이랑 관련 있어?')">이 환자 심부전 관련</div>
      <div class="chip" onclick="fillInput('당뇨병 임상노트 보여줘')">당뇨병 노트</div>
    </div>
  </div>
</div>

<!-- 입력 바 -->
<div id="input-bar">
  <div id="input-wrap">
    <textarea id="user-input" rows="1"
      placeholder="질의를 입력하세요  (Shift+Enter 줄바꿈 / Enter 전송)"></textarea>
    <button id="send-btn" onclick="sendMessage()" title="전송">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
           stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</div>

<div id="toast"></div>

<script>
  const chat     = document.getElementById('chat');
  const input    = document.getElementById('user-input');
  const sendBtn  = document.getElementById('send-btn');
  const counter  = document.getElementById('turn-counter');
  const examples = document.getElementById('examples');
  let   turnCount = 0;
  let   typingRow = null;

  /* ── 자동 높이 조절 ───────────────────────────────────────────── */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });

  /* ── Enter / Shift+Enter ────────────────────────────────────── */
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  /* ── 예시 칩 클릭 ───────────────────────────────────────────── */
  function fillInput(text) {
    input.value = text;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    input.focus();
  }

  /* ── 타임스탬프 ─────────────────────────────────────────────── */
  function now() {
    return new Date().toLocaleTimeString('ko-KR', { hour12: false });
  }

  /* ── 말풍선 추가 ─────────────────────────────────────────────── */
  function appendMsg(role, text) {
    const isUser = role === 'user';
    const row = document.createElement('div');
    row.className = 'msg-row';

    const avatar = document.createElement('div');
    avatar.className = `avatar ${isUser ? 'avatar-user' : 'avatar-agent'}`;
    avatar.textContent = isUser ? '👤' : '⚕';

    const right = document.createElement('div');
    right.style.flex = '1';

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.innerHTML = `<span class="role">${isUser ? 'USER' : 'AGENT'}</span>  ${now()}`;

    const bubble = document.createElement('div');
    bubble.className = `bubble ${isUser ? 'bubble-user' : 'bubble-agent'}`;
    bubble.textContent = text;

    right.appendChild(meta);
    right.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(right);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return bubble;
  }

  /* ── 로딩 도트 ───────────────────────────────────────────────── */
  function showTyping() {
    const row = document.createElement('div');
    row.className = 'msg-row';

    const avatar = document.createElement('div');
    avatar.className = 'avatar avatar-agent';
    avatar.textContent = '⚕';

    const right = document.createElement('div');
    right.style.flex = '1';

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.innerHTML = `<span class="role">AGENT</span>  추론 중...`;

    const typing = document.createElement('div');
    typing.className = 'typing';
    typing.innerHTML = '<span></span><span></span><span></span>';

    right.appendChild(meta);
    right.appendChild(typing);
    row.appendChild(avatar);
    row.appendChild(right);
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    typingRow = row;
  }

  function removeTyping() {
    if (typingRow) { typingRow.remove(); typingRow = null; }
  }

  /* ── 시스템 구분선 ───────────────────────────────────────────── */
  function appendSys(text) {
    const d = document.createElement('div');
    d.className = 'sys-msg';
    d.textContent = text;
    chat.appendChild(d);
  }

  /* ── 에러 토스트 ────────────────────────────────────────────── */
  function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3500);
  }

  /* ── 전송 ───────────────────────────────────────────────────── */
  async function sendMessage() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;

    // 예시 칩 숨기기
    if (examples) examples.style.display = 'none';

    // 입력 초기화
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    appendMsg('user', text);
    appendSys('에이전트 추론 중');
    showTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      removeTyping();

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err.error || `서버 오류 (${res.status})`);
        return;
      }

      const data = await res.json();
      appendMsg('agent', data.report);

      turnCount++;
      counter.textContent = `TURN ${turnCount}`;

    } catch (e) {
      removeTyping();
      showToast('네트워크 오류가 발생했습니다.');
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    # 세션 ID 발급
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    sid = session.get("sid")
    if not sid:
        return jsonify({"error": "세션이 만료되었습니다. 새로고침 해주세요."}), 400

    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "메시지가 비어 있습니다."}), 400

    user_text = data["message"].strip()
    state = get_session_state(sid)

    # 현재 턴 state 구성 (누적 필드 이어받기)
    current_state = AgentState(
        messages=[HumanMessage(content=user_text)],
        conversation_history=state.conversation_history,
        turn_count=state.turn_count,
        last_patient_id=state.last_patient_id,
        last_intent=state.last_intent,
    )

    try:
        final_state = clinical_agent.invoke(current_state)
    except Exception as e:
        return jsonify({"error": f"에이전트 오류: {str(e)}"}), 500

    # 누적 상태 저장
    _sessions[sid] = AgentState(
        messages=final_state.get("messages", []),
        conversation_history=final_state.get("conversation_history", state.conversation_history),
        turn_count=final_state.get("turn_count", state.turn_count),
        last_patient_id=final_state.get("last_patient_id", state.last_patient_id),
        last_intent=final_state.get("last_intent", state.last_intent),
    )

    report = final_state.get("final_report") or "응답을 생성하지 못했습니다."
    return jsonify({"report": report})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
