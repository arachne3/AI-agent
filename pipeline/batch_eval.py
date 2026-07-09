import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
current_dir    = os.path.dirname(os.path.abspath(__file__))
mimic_base_dir = os.path.normpath(os.path.join(current_dir, '..'))

data_dir    = os.path.join(mimic_base_dir, 'physionet.org', 'files', 'mimiciii', '1.4')
newdata_dir = os.path.join(mimic_base_dir, 'newdata')
output_dir  = os.path.join(mimic_base_dir, 'output', 'predictions')
os.makedirs(output_dir, exist_ok=True)

print("="*20 + " [batch_eval.py] 하이브리드 진단 엔진 가동 " + "="*20)
print(f"➔ 임베딩 로드 경로: {newdata_dir}")

# ── STEP 1. 아티팩트 로드 ─────────────────────────────────────────────────────
meta = torch.load(os.path.join(newdata_dir, 'v5_processed_meta.pt'), weights_only=False)
valid_patients = meta["valid_patients"]
top_icd_codes  = meta["top_icd_codes"]

z_patient = torch.load(os.path.join(newdata_dir, 'z_patient.pt'), weights_only=False)
z_disease = torch.load(os.path.join(newdata_dir, 'z_disease.pt'), weights_only=False)

with open(os.path.join(newdata_dir, 'clinical_prompts.json'), 'r', encoding='utf-8') as f:
    id_to_prompt = {item["id"]: item["prompt"] for item in json.load(f)}

# 질병명 사전
df_dicd     = pd.read_csv(
    os.path.join(data_dir, 'D_ICD_DIAGNOSES.csv.gz'),
    usecols=['ICD9_CODE', 'LONG_TITLE'], compression='gzip'
).dropna()
icd_to_name = dict(zip(df_dicd['ICD9_CODE'].astype(str).str.strip(), df_dicd['LONG_TITLE']))

def get_disease_name(code):
    return icd_to_name.get(str(code).strip(), "Unknown Clinical Condition")

# ── STEP 2. Ground Truth 빌드 ─────────────────────────────────────────────────
print(">> [STEP 2] Ground Truth 빌드 중...")
df_diag = pd.read_csv(
    os.path.join(data_dir, 'DIAGNOSES_ICD.csv.gz'),
    usecols=['SUBJECT_ID', 'ICD9_CODE'], compression='gzip'
).dropna()

patient_to_idx = {pid: idx for idx, pid in enumerate(valid_patients)}
icd_to_idx     = {code: idx for idx, code in enumerate(top_icd_codes)}

df_true = df_diag[
    (df_diag['SUBJECT_ID'].isin(patient_to_idx)) &
    (df_diag['ICD9_CODE'].isin(icd_to_idx))
]
patient_true_diseases = {pid: set() for pid in valid_patients}
for _, row in df_true.iterrows():
    patient_true_diseases[int(row['SUBJECT_ID'])].add(str(row['ICD9_CODE']).strip())

# ── STEP 3. 코사인 유사도 전수 평가 ──────────────────────────────────────────
print(">> [STEP 3] 코사인 유사도 전수 평가 중...")

norm_z_p = F.normalize(z_patient, p=2, dim=-1)
norm_z_d = F.normalize(z_disease, p=2, dim=-1)
sim_mat  = torch.mm(norm_z_p, norm_z_d.t())   # [3000, 50]

k_list       = [1, 3, 5]
hit_counts   = {k: 0 for k in k_list}
total_valid  = 0
json_master  = {}

for idx, pid in enumerate(valid_patients):
    true_set = patient_true_diseases[pid]
    if not true_set:
        continue

    total_valid += 1
    p_sims = sim_mat[idx]

    _, top_idxs = torch.topk(p_sims, k=5)
    pred_codes  = [str(top_icd_codes[i.item()]).strip() for i in top_idxs]

    for k in k_list:
        if any(c in true_set for c in pred_codes[:k]):
            hit_counts[k] += 1

    json_master[int(pid)] = {
        "actual_ground_truth": [
            {"icd9_code": c, "disease_name": get_disease_name(c)}
            for c in sorted(true_set)
        ],
        "gnn_top3_predictions": [
            {
                "rank": r + 1,
                "icd9_code": pred_codes[r],
                "disease_name": get_disease_name(pred_codes[r]),
                "cosine_similarity": float(p_sims[top_icd_codes.index(pred_codes[r])].item()),
            }
            for r in range(3)
        ],
    }

print("\n" + "="*23 + " [GNN 예측 성능] " + "="*23)
print(f"➔ 검증 환자: {total_valid:,}명")
for k in k_list:
    print(f"   Hit Rate @ {k} ➔ {hit_counts[k]/total_valid*100:.2f}%")
print("="*60)

with open(os.path.join(newdata_dir, 'newdata_analytics_master.json'), 'w', encoding='utf-8') as f:
    json.dump(json_master, f, ensure_ascii=False, indent=4)
print(f"➔ 전수 분석 DB 저장 완료: {newdata_dir}/newdata_analytics_master.json\n")

# ── STEP 4. LLM 소견서 (상위 5명 샘플) ───────────────────────────────────────
print(">> [STEP 4] LLM 임상 소견서 생성 중 (상위 5명)...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

for pid in valid_patients[:5]:
    p_data   = json_master[pid]
    true_set = patient_true_diseases[pid]

    gnn_hint = "\n".join([
        f"- {p['rank']}순위: ICD-9 [{p['icd9_code']}] ({p['disease_name']}) | 유사도: {p['cosine_similarity']*100:.1f}%"
        for p in p_data["gnn_top3_predictions"]
    ])

    actual_text = ", ".join([
        f"[{g['icd9_code']}] ({g['disease_name']})"
        for g in p_data["actual_ground_truth"]
    ])

    matching = []
    for p in p_data["gnn_top3_predictions"]:
        exist = "존재" if p["icd9_code"] in true_set else "미존재"
        matching.append(f"{p['rank']}순위 '{p['disease_name']}' → 정답 라벨에 {exist}")

    fixed_prefix = (
        f"{'='*57}\n"
        f" [GNN-LLM 하이브리드 분석 명세서 (환자 ID: {pid})]\n"
        f"{'='*57}\n\n"
        f"🟥 [실제 정답 라벨]\n  {actual_text}\n\n"
        f"🟦 [GNN 매칭 결과]\n  " + "\n  ".join(matching) +
        f"\n\n{'-'*57}\n🟩 [임상 추론 AI 소견]\n{'-'*57}\n"
    )

    system_msg = (
        "당신은 ICU 생체 신호 임베딩과 시계열 데이터를 분석하는 전문 임상 추론 AI 입니다.\n\n"
        f"[GNN 공간 매칭 팩트]\n{gnn_hint}\n\n"
        "⚠️ 지침:\n"
        "1. 제공되지 않은 정보(나이, 성별 등)를 추측하여 작성하지 마세요.\n"
        "2. 24시간 바이탈 추세와 예측 질병의 연관성을 전문적으로 분석하세요.\n"
        "3. 제목이나 서두 없이 순수 임상 소견 본문만 작성하세요."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": f"환자 ID: {pid}\n\n{id_to_prompt.get(pid, '바이탈 없음')}"},
            ],
            temperature=0.2,
        )
        report = fixed_prefix + resp.choices[0].message.content

        out_path = os.path.join(output_dir, f"patient_{pid}_hybrid_report.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   ➔ 환자 {pid} 소견서 저장 완료.")
    except Exception as e:
        print(f"   ❌ 환자 {pid} 처리 오류: {e}")

print("\n" + "="*20 + " [완료] batch_eval.py 파이프라인 종료 " + "="*20)
print(f"➔ 소견서 저장 위치: {output_dir}")
