"""
tools.py
--------
Agent 가 호출하는 두 개의 Tool 을 정의한다.

Tool A — gnn_predict_tool   : 환자 ID 를 받아 GNN 임베딩 기반 Top-5 질병 예측
                              + 실제 진단 정답(actual_ground_truth) 함께 반환
Tool B — rag_search_tool    : 질병명 / ICD-9 코드를 받아 임상 노트 RAG 검색

변경사항:
  - [Tool A] newdata_analytics.json 로드 → actual_ground_truth 포함 반환
  - [Tool A] GNN 예측과 정답 간 hit 여부 표시 (matched 필드)
  - [Tool A/B] 전체 try/except 예외처리 강화
  - [Tool B] similarity_search_with_score() 유사도 필터링
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
ANALYTICS_PATH    = os.getenv(
    "ANALYTICS_PATH",
    os.path.join(DATA_DIR, "newdata_analytics_master.json")
)   # ← 정답 데이터
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./output/vector_store")
DICD_PATH         = os.path.join(
    MIMIC_BASE_DIR, "physionet.org", "files", "mimiciii", "1.4", "D_ICD_DIAGNOSES.csv.gz"
)

RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "1.0"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "4"))

# ── 전역 캐시 ──────────────────────────────────────────────────────────────────
_cache: dict = {}


def _load_gnn_artifacts():
    """GNN 임베딩 + 메타데이터 + 정답 데이터를 메모리에 캐시."""
    if "z_patient" in _cache:
        return

    try:
        meta = torch.load(os.path.join(DATA_DIR, "v5_processed_meta.pt"), weights_only=False)
        _cache["valid_patients"] = meta["valid_patients"]
        _cache["top_icd_codes"]  = meta["top_icd_codes"]
        _cache["patient_to_idx"] = {pid: i for i, pid in enumerate(meta["valid_patients"])}

        _cache["z_patient"] = torch.load(os.path.join(DATA_DIR, "z_patient.pt"), weights_only=False)
        _cache["z_disease"] = torch.load(os.path.join(DATA_DIR, "z_disease.pt"), weights_only=False)

        _cache["norm_z_patient"] = F.normalize(_cache["z_patient"], p=2, dim=-1)
        _cache["norm_z_disease"] = F.normalize(_cache["z_disease"], p=2, dim=-1)

        df_dicd = pd.read_csv(
            DICD_PATH, usecols=["ICD9_CODE", "LONG_TITLE"], compression="gzip"
        ).dropna()
        _cache["icd_to_name"] = dict(
            zip(df_dicd["ICD9_CODE"].astype(str).str.strip(), df_dicd["LONG_TITLE"])
        )

        prompt_path = os.path.join(DATA_DIR, "clinical_prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        _cache["id_to_prompt"] = {item["id"]: item["prompt"] for item in prompts}

        # ── 정답 데이터 로드 ──────────────────────────────────────────────────
        analytics_file = _find_analytics_file(ANALYTICS_PATH)
        if analytics_file:
            with open(analytics_file, "r", encoding="utf-8") as f:
                _cache["analytics"] = json.load(f)
            print(f"✅ 정답 데이터 로드 완료: {analytics_file}")
        else:
            _cache["analytics"] = {}
            print("⚠️  정답 데이터 파일을 찾지 못했습니다. 예측만 표시됩니다.")

        print("✅ GNN 아티팩트 캐시 로드 완료.")

    except FileNotFoundError as e:
        raise RuntimeError(f"GNN 아티팩트 파일을 찾을 수 없습니다: {e}") from e
    except Exception as e:
        raise RuntimeError(f"GNN 아티팩트 로드 중 오류: {e}") from e


def _find_analytics_file(path: str) -> str | None:
    """
    analytics 경로가 파일이면 그대로, 디렉터리면 .json 파일을 자동 탐색.
    """
    if os.path.isfile(path):
        return path
    # .json 확장자 붙여보기
    if os.path.isfile(path + ".json"):
        return path + ".json"
    # 디렉터리 안에서 첫 번째 .json 찾기
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith(".json"):
                return os.path.join(path, fname)
    return None


def _load_vector_store():
    """FAISS 벡터 DB 를 메모리에 캐시."""
    if "vector_store" in _cache:
        return

    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"벡터 DB 가 없습니다. rag_builder.py 를 먼저 실행하세요.\n경로: {VECTOR_STORE_PATH}"
        )

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        _cache["vector_store"] = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ FAISS RAG 리트리버 캐시 로드 완료.")

    except Exception as e:
        raise RuntimeError(f"FAISS 벡터 DB 로드 중 오류: {e}") from e


# ── Tool 입력 스키마 ───────────────────────────────────────────────────────────
class GNNPredictInput(BaseModel):
    patient_id: int = Field(description="예측 대상 환자의 SUBJECT_ID (정수)")


class RAGSearchInput(BaseModel):
    query: str = Field(description="검색할 질병명, ICD-9 코드, 또는 임상 키워드")


# ── Tool A: GNN 질병 예측 ──────────────────────────────────────────────────────
@tool("gnn_predict_tool", args_schema=GNNPredictInput)
def gnn_predict_tool(patient_id: int) -> str:
    """
    환자 ID 를 입력받아:
      1. 실제 진단 정답 (actual_ground_truth) 반환
      2. GNN 임베딩 기반 코사인 유사도로 Top-5 예상 질병 반환
      3. 예측과 정답 간 일치 여부(matched) 표시
    결과는 JSON 문자열로 반환된다.
    """
    try:
        _load_gnn_artifacts()
    except RuntimeError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    try:
        patient_to_idx = _cache["patient_to_idx"]
        if patient_id not in patient_to_idx:
            return json.dumps(
                {"error": f"환자 ID {patient_id} 는 코호트에 없습니다."},
                ensure_ascii=False,
            )

        # ── GNN 예측 ──────────────────────────────────────────────────────────
        idx   = patient_to_idx[patient_id]
        p_vec = _cache["norm_z_patient"][idx]
        sims  = torch.mv(_cache["norm_z_disease"], p_vec)

        top_vals, top_idxs = torch.topk(sims, k=5)
        top_icd_codes = _cache["top_icd_codes"]
        icd_to_name   = _cache["icd_to_name"]

        # ── 정답 ICD 코드 집합 ────────────────────────────────────────────────
        analytics     = _cache.get("analytics", {})
        patient_data  = analytics.get(str(patient_id), {})
        ground_truth  = patient_data.get("actual_ground_truth", [])
        gt_icd_set    = {item["icd9_code"] for item in ground_truth}

        # ── 예측 결과 + hit 표시 ──────────────────────────────────────────────
        predictions = []
        for rank, (val, didx) in enumerate(zip(top_vals.tolist(), top_idxs.tolist()), start=1):
            code = str(top_icd_codes[didx]).strip()
            predictions.append({
                "rank": rank,
                "icd9_code": code,
                "disease_name": icd_to_name.get(code, "Unknown"),
                "cosine_similarity": round(val, 4),
                "matched": code in gt_icd_set,   # ← 정답과 일치 여부
            })

        vital_summary = _cache["id_to_prompt"].get(patient_id, "바이탈 요약 없음")

        result = {
            "patient_id": patient_id,
            "actual_ground_truth": ground_truth,   # ← 실제 진단 정답
            "top5_predictions": predictions,        # ← GNN 예측 (matched 포함)
            "vital_summary": vital_summary,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps(
            {"error": f"GNN 예측 중 오류가 발생했습니다: {e}"},
            ensure_ascii=False,
        )


# ── Tool B: RAG 임상 노트 검색 ────────────────────────────────────────────────
@tool("rag_search_tool", args_schema=RAGSearchInput)
def rag_search_tool(query: str) -> str:
    """
    질병명, ICD-9 코드, 또는 임상 키워드로 MIMIC-III Discharge summary 를 검색하고
    관련 임상 노트 발췌문을 반환한다.
    """
    try:
        _load_vector_store()
    except (FileNotFoundError, RuntimeError) as e:
        return f"[벡터 DB 오류] {e}"

    try:
        vs = _cache["vector_store"]
        docs_with_scores = vs.similarity_search_with_score(query, k=RAG_TOP_K)

        if not docs_with_scores:
            return "관련 임상 노트를 찾지 못했습니다."

        filtered = [
            (doc, score)
            for doc, score in docs_with_scores
            if score <= RAG_SCORE_THRESHOLD
        ]

        if not filtered:
            best_score = docs_with_scores[0][1]
            return (
                f"관련 임상 노트를 찾지 못했습니다. "
                f"(검색어: '{query}' | 최고 유사도 점수: {best_score:.4f} > 임계값 {RAG_SCORE_THRESHOLD})\n"
                f"더 구체적인 질병명이나 영문 의학 용어로 다시 질문해 보세요."
            )

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

    except Exception as e:
        return f"[RAG 검색 오류] 검색 중 문제가 발생했습니다: {e}"


TOOLS = [gnn_predict_tool, rag_search_tool]
