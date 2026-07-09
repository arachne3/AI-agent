import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
# pipeline/ 폴더의 부모 = AI Agent 루트
current_dir   = os.path.dirname(os.path.abspath(__file__))
mimic_base_dir = os.path.normpath(os.path.join(current_dir, '..'))

data_dir    = os.path.join(mimic_base_dir, 'physionet.org', 'files', 'mimiciii', '1.4')
newdata_dir = os.path.join(mimic_base_dir, 'newdata')
output_dir  = os.path.join(mimic_base_dir, 'output', 'predictions')

os.makedirs(newdata_dir, exist_ok=True)
os.makedirs(output_dir,  exist_ok=True)

print("="*20 + " [load.py] 다중 질병 균형 분포 코호트(3,000명) 빌더 가동 " + "="*20)
print(f"➔ 원천 데이터 경로: {data_dir}")
print(f"➔ 산출물 저장 경로: {newdata_dir}")

# ── 바이탈 매핑 ───────────────────────────────────────────────────────────────
VITAL_MAPPING = {
    "heart_rate": {"ids": [211, 220045],                          "min": 30,  "max": 250},
    "sys_bp":     {"ids": [51, 442, 455, 220179, 220050],         "min": 40,  "max": 300},
    "dia_bp":     {"ids": [8368, 8440, 8441, 220180, 220051],     "min": 20,  "max": 200},
    "resprate":   {"ids": [618, 220210],                          "min": 5,   "max": 80},
    "spo2":       {"ids": [646, 220277],                          "min": 50,  "max": 100},
    "temp":       {"ids": [676, 677, 223762, 223761],             "min": 30,  "max": 45},
}

itemid_to_vital = {}
for vital_name, meta in VITAL_MAPPING.items():
    for itemid in meta["ids"]:
        itemid_to_vital[itemid] = vital_name

def normalize_temp(val):
    if val > 90:
        return (val - 32) * 5 / 9
    return val

# ── STEP 1. 층화 균형 샘플링 ──────────────────────────────────────────────────
print(">> [STEP 1] 균형 분포 기반 후보군 선별...")
diagnoses_path = os.path.join(data_dir, 'DIAGNOSES_ICD.csv.gz')
if not os.path.exists(diagnoses_path):
    raise FileNotFoundError(f"파일 없음: {diagnoses_path}")

df_diag = pd.read_csv(diagnoses_path, usecols=['SUBJECT_ID', 'ICD9_CODE'],
                      compression='gzip').dropna()

top_icd_codes      = df_diag['ICD9_CODE'].value_counts().head(50).index.tolist()
candidate_patients = set()
target_per_disease = 150

for icd in top_icd_codes:
    pids = df_diag[df_diag['ICD9_CODE'] == icd]['SUBJECT_ID'].unique()
    sampled = np.random.choice(pids, size=min(len(pids), target_per_disease), replace=False)
    candidate_patients.update(sampled)

candidate_patients = list(candidate_patients)
print(f"➔ 1차 후보 환자군: {len(candidate_patients):,}명")

# ── STEP 2. CHARTEVENTS 청크 스트리밍 ────────────────────────────────────────
print(">> [STEP 2] CHARTEVENTS 스트리밍 시작...")
chartevents_path = os.path.join(data_dir, 'CHARTEVENTS.csv.gz')
if not os.path.exists(chartevents_path):
    raise FileNotFoundError(f"파일 없음: {chartevents_path}")

patient_vital_db = {}
candidate_set    = set(candidate_patients)
chunk_size       = 400_000
total_rows       = 0

for chunk in pd.read_csv(chartevents_path,
                          usecols=['SUBJECT_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
                          chunksize=chunk_size, compression='gzip'):
    filtered = chunk[
        (chunk['SUBJECT_ID'].isin(candidate_set)) &
        (chunk['ITEMID'].isin(itemid_to_vital.keys()))
    ].dropna()

    for _, row in filtered.iterrows():
        pid        = int(row['SUBJECT_ID'])
        vital_name = itemid_to_vital[int(row['ITEMID'])]
        val        = float(row['VALUENUM'])
        meta       = VITAL_MAPPING[vital_name]

        if vital_name == "temp":
            val = normalize_temp(val)
        if not (meta["min"] <= val <= meta["max"]):
            continue

        if pid not in patient_vital_db:
            patient_vital_db[pid] = {v: [] for v in VITAL_MAPPING}
        patient_vital_db[pid][vital_name].append((row['CHARTTIME'], val))

    total_rows += chunk_size
    if total_rows % 4_000_000 == 0:
        print(f"   ➔ {total_rows:,} 행 처리 완료...")

# ── STEP 3. 24H 슬라이딩 윈도우 검증 → 3,000명 확정 ─────────────────────────
print(">> [STEP 3] 24H 시계열 검증 및 최종 3,000명 확정...")
valid_cohort          = []
clinical_prompts_master = []

for pid, vitals in tqdm(patient_vital_db.items(), desc="24H 검증"):
    if len(valid_cohort) >= 3000:
        break
    if any(len(vitals[v]) == 0 for v in VITAL_MAPPING):
        continue

    all_times = []
    for v in VITAL_MAPPING:
        all_times.extend([pd.to_datetime(t[0]) for t in vitals[v]])

    last_time   = max(all_times)
    start_24h   = last_time - pd.Timedelta(hours=24)
    intervals   = [start_24h + pd.Timedelta(hours=6*i) for i in range(5)]
    chunks_dict = {v: [[] for _ in range(4)] for v in VITAL_MAPPING}

    for v in VITAL_MAPPING:
        for ctime_str, val in vitals[v]:
            t_dt = pd.to_datetime(ctime_str)
            if start_24h <= t_dt <= last_time:
                for i in range(4):
                    if intervals[i] <= t_dt < intervals[i+1]:
                        chunks_dict[v][i].append(val)
                        break

    if any(any(len(chunks_dict[v][i]) == 0 for i in range(4)) for v in VITAL_MAPPING):
        continue

    valid_cohort.append(pid)

    prompt = "[최근 24시간 6시간 간격 바이탈 사인 시계열]\n"
    for i in range(4):
        prompt += f"  - {24-6*i}h~{24-6*(i+1)}h 전 ➔ "
        prompt += " | ".join(
            f"평균 {v}: {np.mean(chunks_dict[v][i]):.1f}" for v in VITAL_MAPPING
        ) + "\n"

    clinical_prompts_master.append({"id": pid, "prompt": prompt})

print(f"➔ 최종 코호트: {len(valid_cohort)}명")

# ── STEP 4. 피처 텐서 생성 ───────────────────────────────────────────────────
print(">> [STEP 4] 환자/질병 피처 텐서 빌드...")

patient_features_list = []
for pid in valid_cohort:
    feat = []
    for v in VITAL_MAPPING:
        vals = [t[1] for t in patient_vital_db[pid][v]]
        feat.extend([np.mean(vals), np.std(vals)])
    patient_features_list.append(feat)
X_patient = torch.tensor(patient_features_list, dtype=torch.float32)

disease_features_list = []
for icd in top_icd_codes:
    shared_pids  = df_diag[df_diag['ICD9_CODE'] == icd]['SUBJECT_ID'].unique()
    icd_vitals   = {v: [] for v in VITAL_MAPPING}
    for spid in shared_pids:
        if spid in patient_vital_db:
            for v in VITAL_MAPPING:
                icd_vitals[v].extend([t[1] for t in patient_vital_db[spid][v]])
    centroid = [np.mean(icd_vitals[v]) if icd_vitals[v] else 0.0 for v in VITAL_MAPPING]
    disease_features_list.append(centroid)
X_disease = torch.tensor(disease_features_list, dtype=torch.float32)

# ── STEP 5. 저장 ─────────────────────────────────────────────────────────────
print(">> [STEP 5] 산출물 저장 중...")

with open(os.path.join(newdata_dir, 'clinical_prompts.json'), 'w', encoding='utf-8') as f:
    json.dump(clinical_prompts_master, f, ensure_ascii=False, indent=4)

torch.save({
    "valid_patients": valid_cohort,
    "top_icd_codes":  top_icd_codes,
    "X_patient":      X_patient,
    "X_disease":      X_disease,
}, os.path.join(newdata_dir, 'v5_processed_meta.pt'))

print("="*20 + " [완료] newdata/ 에 저장 완료 " + "="*20)
print(f"➔ {newdata_dir}")
