"""
rag_builder.py
--------------
MIMIC-III NOTEEVENTS 에서 기존 3,000명 코호트의 Discharge summary 를 추출하고
FAISS 벡터 DB 로 인덱싱한다.

실행 방법:
    python rag/rag_builder.py
    
산출물:
    output/vector_store/   ← FAISS 인덱스 + 메타데이터
"""

import os
import torch
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
MIMIC_BASE_DIR   = os.getenv("MIMIC_BASE_DIR", "C:/Users/Arachne/Desktop/MIMIC")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./output/vector_store")
DATA_DIR         = os.path.join(MIMIC_BASE_DIR, "newdata")
NOTES_PATH       = os.path.join(MIMIC_BASE_DIR, "physionet.org", "files", "mimiciii", "1.4",
                                "NOTEEVENTS.csv", "NOTEEVENTS.csv")

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


def load_cohort_ids() -> set:
    """load.py 가 저장한 메타번들에서 3,000명 SUBJECT_ID 를 가져온다."""
    meta_path = os.path.join(DATA_DIR, "v5_processed_meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"v5_processed_meta.pt 가 없습니다. load.py 를 먼저 실행하세요.\n경로: {meta_path}"
        )
    meta = torch.load(meta_path, weights_only=False)
    return set(meta["valid_patients"])


def extract_discharge_summaries(cohort_ids: set) -> list[Document]:
    """
    NOTEEVENTS 에서 코호트 환자의 Discharge summary 만 추출한다.
    - 3.8 GB 파일을 청크 스트리밍으로 처리
    - ISERROR == 1 인 노트는 제외
    """
    print(f">> NOTEEVENTS 스트리밍 시작 (코호트 {len(cohort_ids):,}명 대상)...")

    documents: list[Document] = []
    chunk_size = 50_000
    processed_rows = 0

    for chunk in pd.read_csv(
        NOTES_PATH,
        usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT", "ISERROR"],
        chunksize=chunk_size,
        low_memory=False,
    ):
        # 코호트 + Discharge summary + 에러 없음 필터
        filtered = chunk[
            (chunk["SUBJECT_ID"].isin(cohort_ids))
            & (chunk["CATEGORY"] == "Discharge summary")
            & (chunk["ISERROR"].fillna(0).astype(str) != "1")
        ].dropna(subset=["TEXT"])

        for _, row in filtered.iterrows():
            text = str(row["TEXT"]).strip()
            if len(text) < 100:   # 너무 짧은 노트 제외
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "subject_id": int(row["SUBJECT_ID"]),
                        "hadm_id": int(row["HADM_ID"]) if pd.notna(row["HADM_ID"]) else -1,
                        "category": row["CATEGORY"],
                    },
                )
            )

        processed_rows += chunk_size
        if processed_rows % 500_000 == 0:
            print(f"   ➔ {processed_rows:,} 행 처리 완료 | 수집 문서: {len(documents):,}건")

    print(f">> 추출 완료: Discharge summary {len(documents):,}건")
    return documents


def build_vector_store(documents: list[Document]) -> FAISS:
    """
    문서를 청크 분할 후 OpenAI 임베딩으로 FAISS 인덱스를 구축한다.
    """
    print(">> 텍스트 청크 분할 중...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"   ➔ 총 청크 수: {len(chunks):,}개")

    print(">> OpenAI 임베딩 벡터 DB 구축 중 (시간이 걸릴 수 있습니다)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # 비용 효율 모델
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 배치 단위로 나눠서 임베딩 (rate limit 방어)
    BATCH = 500
    all_chunks_batched = [chunks[i : i + BATCH] for i in range(0, len(chunks), BATCH)]

    vector_store = None
    for i, batch in enumerate(tqdm(all_chunks_batched, desc="임베딩 배치")):
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

    return vector_store


def main():
    print("=" * 60)
    print("  RAG Builder — NOTEEVENTS → FAISS Vector Store")
    print("=" * 60)

    cohort_ids   = load_cohort_ids()
    documents    = extract_discharge_summaries(cohort_ids)

    if not documents:
        raise ValueError("추출된 문서가 없습니다. NOTEEVENTS 경로와 코호트 ID 를 확인하세요.")

    vector_store = build_vector_store(documents)
    vector_store.save_local(VECTOR_STORE_PATH)

    print(f"\n✅ FAISS 벡터 DB 저장 완료 → {VECTOR_STORE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
