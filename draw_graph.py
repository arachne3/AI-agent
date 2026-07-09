"""
draw_graph.py
-------------
LangGraph Mermaid 다이어그램 출력 스크립트.
프로젝트 루트(AI Agent/)에서 실행:
    python draw_graph.py
"""

from agent.graph import clinical_agent

print(clinical_agent.get_graph().draw_mermaid())
