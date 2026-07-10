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
import os, uuid

from agent.graph import clinical_agent
from agent.state import AgentState
from middleware.middleware import logger

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)

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
  :root {
    --bg-base:     #0b0f1a;
    --bg-panel:    #111827;
    --bg-input:    #1a2236;
    --bg-bubble-u: #1e2d45;
    --bg-bubble-a: #131c2e;
    --accent:      #38bdf8;
    --accent-dim:  #0ea5e9;
    --warn:        #f59e0b;
    --ok:          #34d399;
    --err:         #f87171;
    --text-pri:    #e2e8f0;
    --text-sec:    #94a3b8;
    --text-dim:    #475569;
    --border:      #1e3a5f;
    --sidebar-w:   300px;
    --radius:      10px;
    --font-mono:   'JetBrains Mono','Fira Code','Consolas',monospace;
    --font-body:   'Inter','Pretendard',system-ui,sans-serif;
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

  /* ── 헤더 ──────────────────────────────────────────────────── */
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    z-index: 10;
  }

  /* 햄버거 버튼 */
  #hamburger {
    width: 36px; height: 36px;
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 5px;
    flex-shrink: 0;
    transition: border-color 0.15s;
  }
  #hamburger:hover { border-color: var(--accent); }
  #hamburger span {
    display: block;
    width: 16px; height: 1.5px;
    background: var(--text-sec);
    border-radius: 2px;
    transition: background 0.15s, transform 0.25s, opacity 0.25s;
  }
  /* X 애니메이션 */
  body.sidebar-open #hamburger span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
  body.sidebar-open #hamburger span:nth-child(2) { opacity: 0; }
  body.sidebar-open #hamburger span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }

  .logo-pulse {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    animation: pulse 2.4s ease-in-out infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%,100% { opacity:1; box-shadow:0 0 8px var(--ok); }
    50%      { opacity:0.4; box-shadow:0 0 2px var(--ok); }
  }

  header h1 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border);
    color: var(--text-sec);
  }
  header .sub {
    font-size: 11px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    margin-left: auto;
  }

  /* ── 레이아웃: 사이드바 + 메인 ─────────────────────────────── */
  #layout {
    flex: 1;
    display: flex;
    overflow: hidden;
    position: relative;
  }

  /* ── 오버레이 (모바일) ─────────────────────────────────────── */
  #overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.5);
    z-index: 19;
  }
  body.sidebar-open #overlay { display: block; }

  /* ── 사이드바 ───────────────────────────────────────────────── */
  #sidebar {
    width: var(--sidebar-w);
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    transform: translateX(calc(-1 * var(--sidebar-w)));
    transition: transform 0.28s cubic-bezier(0.4,0,0.2,1);
    position: absolute;
    top: 0; left: 0; bottom: 0;
    z-index: 20;
  }
  body.sidebar-open #sidebar { transform: translateX(0); }

  .sidebar-header {
    padding: 16px 16px 12px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .sidebar-header p {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  /* 검색 */
  #patient-search {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 7px;
    color: var(--text-pri);
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 8px 12px;
    outline: none;
    transition: border-color 0.15s;
  }
  #patient-search::placeholder { color: var(--text-dim); }
  #patient-search:focus { border-color: var(--accent); }

  /* 환자 목록 */
  #patient-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  #patient-list::-webkit-scrollbar { width: 3px; }
  #patient-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* 로딩 / 에러 */
  .list-status {
    padding: 24px 12px;
    text-align: center;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-dim);
  }

  /* 환자 카드 */
  .patient-card {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .patient-card:hover {
    background: var(--bg-input);
    border-color: var(--border);
  }
  .patient-card.active {
    background: #0c1f35;
    border-color: var(--accent);
  }

  .patient-icon {
    width: 28px; height: 28px;
    border-radius: 6px;
    background: #0c1f35;
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
  }
  .patient-info { flex: 1; min-width: 0; }
  .patient-id {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
  }
  .patient-sub {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 1px;
  }

  /* 페이지네이션 */
  #pagination {
    padding: 10px 8px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-shrink: 0;
  }
  .page-btn {
    flex: 1;
    padding: 7px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 7px;
    color: var(--text-sec);
    font-size: 11px;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    text-align: center;
  }
  .page-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .page-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  #page-info {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    white-space: nowrap;
  }

  /* ── 메인 채팅 영역 ─────────────────────────────────────────── */
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    /* 사이드바가 absolute 라 메인은 항상 전체 너비 */
  }

  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    scroll-behavior: smooth;
  }
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg-row {
    display: flex;
    gap: 12px;
    padding: 6px 24px;
    max-width: 860px;
    margin: 0 auto;
    width: 100%;
  }

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

  .bubble {
    flex: 1;
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.7;
    font-size: 13.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .bubble-user  { background: var(--bg-bubble-u); border: 1px solid var(--border); }
  .bubble-agent { background: var(--bg-bubble-a); border: 1px solid #1a3050; }

  .msg-meta {
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-dim);
    margin-bottom: 4px;
  }
  .msg-meta .role { color: var(--accent); }

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
  .sys-msg::before { content:''; display:block; width:24px; height:1px; background:var(--border); }
  .sys-msg::after  { content:''; display:block; flex:1; height:1px; background:var(--border); }

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
    0%,80%,100% { transform:scale(0.7); opacity:0.2; }
    40%         { transform:scale(1.1); opacity:1; }
  }

  /* 예시 칩 */
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
  .chip:hover { border-color: var(--accent); color: var(--accent); }

  /* 입력 바 */
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

  /* 토스트 */
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
  #toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

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
  <button id="hamburger" onclick="toggleSidebar()" title="환자 목록">
    <span></span><span></span><span></span>
  </button>
  <div class="logo-pulse"></div>
  <h1>ClinicalGNN-LLM Agent</h1>
  <span class="badge">MIMIC-III</span>
  <span class="sub" id="turn-counter">TURN 0</span>
</header>

<!-- 레이아웃 -->
<div id="layout">

  <!-- 오버레이 -->
  <div id="overlay" onclick="closeSidebar()"></div>

  <!-- 사이드바 -->
  <aside id="sidebar">
    <div class="sidebar-header">
      <p>🏥 환자 목록</p>
      <input id="patient-search" type="text"
             placeholder="ID 검색..."
             oninput="filterPatients(this.value)">
    </div>
    <div id="patient-list">
      <div class="list-status">로딩 중...</div>
    </div>
    <div id="pagination">
      <button class="page-btn" id="prev-btn" onclick="changePage(-1)" disabled>← 이전</button>
      <span id="page-info">-</span>
      <button class="page-btn" id="next-btn" onclick="changePage(1)" disabled>다음 →</button>
    </div>
  </aside>

  <!-- 메인 채팅 -->
  <div id="main">
    <div id="chat">
      <div id="examples">
        <p>예시 질의</p>
        <div class="chip-row">
          <div class="chip" onclick="fillInput('환자 1197번 예측해줘')">환자 1197번 예측</div>
          <div class="chip" onclick="fillInput('폐렴에 대해 임상노트 검색해줘')">폐렴 검색</div>
          <div class="chip" onclick="fillInput('환자 10006번 예측하고 질병 설명도 해줘')">예측 + 설명</div>
          <div class="chip" onclick="fillInput('이 환자 심부전이랑 관련 있어?')">멀티턴 질의</div>
        </div>
      </div>
    </div>

    <div id="input-bar">
      <div id="input-wrap">
        <textarea id="user-input" rows="1"
          placeholder="질의를 입력하세요  (Shift+Enter 줄바꿈 / Enter 전송)"></textarea>
        <button id="send-btn" onclick="sendMessage()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"
               stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
  /* ── 상태 ──────────────────────────────────────────────────── */
  const chat      = document.getElementById('chat');
  const input     = document.getElementById('user-input');
  const sendBtn   = document.getElementById('send-btn');
  const counter   = document.getElementById('turn-counter');
  const examples  = document.getElementById('examples');
  let   turnCount = 0;
  let   typingRow = null;

  /* ── 사이드바 상태 ─────────────────────────────────────────── */
  let allPatients   = [];   // 전체 환자 ID 배열
  let filteredList  = [];   // 검색 필터 후 목록
  let currentPage   = 0;
  const PAGE_SIZE   = 15;
  let activeId      = null;

  /* ── 사이드바 토글 ─────────────────────────────────────────── */
  function toggleSidebar() {
    const open = document.body.classList.toggle('sidebar-open');
    if (open && allPatients.length === 0) loadPatients();
  }
  function closeSidebar() {
    document.body.classList.remove('sidebar-open');
  }

  /* ── 환자 목록 로드 ────────────────────────────────────────── */
  async function loadPatients() {
    const listEl = document.getElementById('patient-list');
    listEl.innerHTML = '<div class="list-status">불러오는 중...</div>';
    try {
      const res  = await fetch('/patients');
      const data = await res.json();
      allPatients  = data.patients || [];
      filteredList = [...allPatients];
      currentPage  = 0;
      renderPage();
    } catch(e) {
      listEl.innerHTML = '<div class="list-status" style="color:var(--err)">목록 로드 실패</div>';
    }
  }

  /* ── 검색 필터 ─────────────────────────────────────────────── */
  function filterPatients(q) {
    const s = q.trim();
    filteredList = s
      ? allPatients.filter(id => String(id).includes(s))
      : [...allPatients];
    currentPage = 0;
    renderPage();
  }

  /* ── 페이지 렌더 ───────────────────────────────────────────── */
  function renderPage() {
    const listEl   = document.getElementById('patient-list');
    const total    = filteredList.length;
    const totalPg  = Math.ceil(total / PAGE_SIZE) || 1;
    const start    = currentPage * PAGE_SIZE;
    const slice    = filteredList.slice(start, start + PAGE_SIZE);

    document.getElementById('page-info').textContent =
      `${currentPage + 1} / ${totalPg}  (총 ${total}명)`;
    document.getElementById('prev-btn').disabled = currentPage === 0;
    document.getElementById('next-btn').disabled = currentPage >= totalPg - 1;

    if (slice.length === 0) {
      listEl.innerHTML = '<div class="list-status">검색 결과 없음</div>';
      return;
    }

    listEl.innerHTML = '';
    slice.forEach(pid => {
      const card = document.createElement('div');
      card.className = 'patient-card' + (pid === activeId ? ' active' : '');
      card.innerHTML = `
        <div class="patient-icon">🧑</div>
        <div class="patient-info">
          <div class="patient-id">ID · ${pid}</div>
          <div class="patient-sub">클릭하여 예측 질의</div>
        </div>`;
      card.onclick = () => selectPatient(pid);
      listEl.appendChild(card);
    });
  }

  /* ── 페이지 이동 ───────────────────────────────────────────── */
  function changePage(delta) {
    const totalPg = Math.ceil(filteredList.length / PAGE_SIZE) || 1;
    currentPage = Math.max(0, Math.min(currentPage + delta, totalPg - 1));
    renderPage();
  }

  /* ── 환자 선택 → 입력창 자동 채움 ─────────────────────────── */
  function selectPatient(pid) {
    activeId = pid;
    renderPage();   // active 스타일 반영
    fillInput(`환자 ${pid}번 예측해줘`);
    closeSidebar();
  }

  /* ── 예시 칩 클릭 ──────────────────────────────────────────── */
  function fillInput(text) {
    input.value = text;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    input.focus();
  }

  /* ── 자동 높이 ─────────────────────────────────────────────── */
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  /* ── 유틸 ──────────────────────────────────────────────────── */
  function now() {
    return new Date().toLocaleTimeString('ko-KR', { hour12: false });
  }
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
  }
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
  function appendSys(text) {
    const d = document.createElement('div');
    d.className = 'sys-msg';
    d.textContent = text;
    chat.appendChild(d);
  }
  function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3500);
  }

  /* ── 전송 ──────────────────────────────────────────────────── */
  async function sendMessage() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;
    if (examples) examples.style.display = 'none';
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
    } catch(e) {
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
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return render_template_string(HTML)


@app.route("/patients")
def patients():
    """환자 ID 목록 반환 — 사이드바에서 호출"""
    try:
        from tools.tools import _load_gnn_artifacts, _cache
        _load_gnn_artifacts()
        patient_list = [int(p) for p in _cache["valid_patients"]]
        return jsonify({"patients": patient_list, "total": len(patient_list)})
    except Exception as e:
        logger.error(f"patients endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


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
