"""
tools.py
--------
Agent 가 호출하는 두 개의 Tool 을 정의한다.

Tool A — gnn_predict_tool   : 환자 ID 를 받아 GNN 임베딩 기반 Top-5 질병 예측
Tool B — rag_search_tool    : 질병명 / ICD-9 코드를 받아 임상 노트 RAG 검색

변경사항:
  - [Tool B] as_retriever() → similarity_search_with_score() 로 교체
    → 유사도 점수 기반 필터링 적용 (RAG_SCORE_THRESHOLD 미만 결과 제거)
  - [Tool B] 검색 결과에 유사도 점수 표시 추가 (디버깅 및 report_node 판단용)
  - [Tool A] vital_summary 원문 전달 유지 (graph.py 에서 system prompt 로 처리)
  - [공통]  캐시 키 충돌 방지를 위해 _cache 접근 일원화
"""

import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
MIMIC_BASE_DIR    = os.getenv("MIMIC_BASE_DIR", "C:/Users/Arachne/Desktop/MIMIC")
DATA_DIR          = os.path.join(MIMIC_BASE_DIR, "newdata")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./output/vector_store")
DICD_PATH         = os.path.join(
    MIMIC_BASE_DIR, "physionet.org", "files", "mimiciii", "1.4", "D_ICD_DIAGNOSES.csv.gz"
)

# RAG 유사도 임계값: 이 값 미만의 결과는 "무관한 노트"로 판단하여 제외
# FAISS L2 distance 기반 → 낮을수록 유사 (0에 가까울수록 완벽 매치)
# 실험적으로 1.0 이하를 "관련 있음"으로 설정 (조정 가능)
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "1.0"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "4"))

# ── 전역 캐시 (모듈 로드 시 1회만 로드) ──────────────────────────────────────
_cache: dict = {}


def _load_gnn_artifacts():
    """GNN 임베딩 + 메타데이터를 메모리에 캐시."""
    if "z_patient" in _cache:
        return

    meta = torch.load(os.path.join(DATA_DIR, "v5_processed_meta.pt"), weights_only=False)
    _cache["valid_patients"] = meta["valid_patients"]
    _cache["top_icd_codes"]  = meta["top_icd_codes"]
    _cache["patient_to_idx"] = {pid: i for i, pid in enumerate(meta["valid_patients"])}

    _cache["z_patient"] = torch.load(os.path.join(DATA_DIR, "z_patient.pt"), weights_only=False)
    _cache["z_disease"] = torch.load(os.path.join(DATA_DIR, "z_disease.pt"), weights_only=False)

    # 정규화된 행렬 미리 계산
    _cache["norm_z_patient"] = F.normalize(_cache["z_patient"], p=2, dim=-1)
    _cache["norm_z_disease"] = F.normalize(_cache["z_disease"], p=2, dim=-1)

    # 질병명 사전
    df_dicd = pd.read_csv(DICD_PATH, usecols=["ICD9_CODE", "LONG_TITLE"], compression="gzip").dropna()
    _cache["icd_to_name"] = dict(
        zip(df_dicd["ICD9_CODE"].astype(str).str.strip(), df_dicd["LONG_TITLE"])
    )

    # 임상 프롬프트 (바이탈 원문)
    prompt_path = os.path.join(DATA_DIR, "clinical_prompts.json")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    _cache["id_to_prompt"] = {item["id"]: item["prompt"] for item in prompts}

    print("✅ GNN 아티팩트 캐시 로드 완료.")


def _load_vector_store():
    """FAISS 벡터 DB 를 메모리에 캐시. (retriever 가 아닌 vectorstore 객체로 저장)"""
    if "vector_store" in _cache:
        return

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"벡터 DB 가 없습니다. rag_builder.py 를 먼저 실행하세요.\n경로: {VECTOR_STORE_PATH}"
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    # ✅ retriever 대신 vectorstore 원본 저장 → score 접근 가능
    _cache["vector_store"] = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ FAISS RAG 리트리버 캐시 로드 완료.")


# ── Tool A 입력 스키마 ─────────────────────────────────────────────────────────
class GNNPredictInput(BaseModel):
    patient_id: int = Field(description="예측 대상 환자의 SUBJECT_ID (정수)")


# ── Tool B 입력 스키마 ─────────────────────────────────────────────────────────
class RAGSearchInput(BaseModel):
    query: str = Field(description="검색할 질병명, ICD-9 코드, 또는 임상 키워드")


# ── Tool A: GNN 질병 예측 ──────────────────────────────────────────────────────
@tool("gnn_predict_tool", args_schema=GNNPredictInput)
def gnn_predict_tool(patient_id: int) -> str:
    """
    환자 ID 를 입력받아 GNN 임베딩 기반 코사인 유사도로 Top-5 예상 질병을 반환한다.
    결과는 JSON 문자열로 반환된다.
    """
    _load_gnn_artifacts()

    patient_to_idx = _cache["patient_to_idx"]
    if patient_id not in patient_to_idx:
        return json.dumps(
            {"error": f"환자 ID {patient_id} 는 코호트에 없습니다."},
            ensure_ascii=False,
        )

    idx   = patient_to_idx[patient_id]
    p_vec = _cache["norm_z_patient"][idx]           # [128]
    sims  = torch.mv(_cache["norm_z_disease"], p_vec)  # [50]

    top_vals, top_idxs = torch.topk(sims, k=5)
    top_icd_codes = _cache["top_icd_codes"]
    icd_to_name   = _cache["icd_to_name"]

    predictions = []
    for rank, (val, didx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist()), start=1):
        code = str(top_icd_codes[didx]).strip()
        predictions.append({
            "rank": rank,
            "icd9_code": code,
            "disease_name": icd_to_name.get(code, "Unknown"),
            "cosine_similarity": round(val, 4),
        })

    # vital_summary: 원문 그대로 전달 (graph.py report_node 의 system prompt 에서 처리)
    vital_summary = _cache["id_to_prompt"].get(patient_id, "바이탈 요약 없음")

    result = {
        "patient_id": patient_id,
        "top5_predictions": predictions,
        "vital_summary": vital_summary,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Tool B: RAG 임상 노트 검색 ────────────────────────────────────────────────
@tool("rag_search_tool", args_schema=RAGSearchInput)
def rag_search_tool(query: str) -> str:
    """
    질병명, ICD-9 코드, 또는 임상 키워드로 MIMIC-III Discharge summary 를 검색하고
    관련 임상 노트 발췌문을 반환한다.

    유사도 필터:
      - FAISS L2 distance 가 RAG_SCORE_THRESHOLD(기본 1.0) 초과인 문서는 제외
      - 필터링 후 남은 문서가 없으면 "관련 노트 없음" 메시지 반환
    """
    _load_vector_store()

    vs = _cache["vector_store"]

    # ✅ similarity_search_with_score: (Document, L2_distance) 튜플 반환
    # L2 distance: 낮을수록 유사 (0 = 완벽 일치)
    docs_with_scores = vs.similarity_search_with_score(query, k=RAG_TOP_K)

    if not docs_with_scores:
        return "관련 임상 노트를 찾지 못했습니다."

    # ── 유사도 필터링 ─────────────────────────────────────────────────────────
    filtered = [
        (doc, score)
        for doc, score in docs_with_scores
        if score <= RAG_SCORE_THRESHOLD
    ]

    if not filtered:
        best_score = docs_with_scores[0][1] if docs_with_scores else "N/A"
        return (
            f"관련 임상 노트를 찾지 못했습니다. "
            f"(검색어: '{query}' | 최고 유사도 점수: {best_score:.4f} > 임계값 {RAG_SCORE_THRESHOLD})\n"
            f"더 구체적인 질병명이나 영문 의학 용어로 다시 질문해 보세요."
        )

    # ── 결과 포맷팅 (점수 포함) ───────────────────────────────────────────────
    results = []
    for i, (doc, score) in enumerate(filtered, start=1):
        meta    = doc.metadata
        snippet = doc.page_content[:600].replace("\n", " ")
        results.append(
            f"[참고 {i}] 환자 {meta.get('subject_id', '?')} "
            f"(입원 {meta.get('hadm_id', '?')}) | 유사도 점수: {score:.4f}\n"
            f"{snippet}..."
        )

    return "\n\n".join(results)


# 외부에서 import 할 수 있도록 목록 노출
TOOLS = [gnn_predict_tool, rag_search_tool]
