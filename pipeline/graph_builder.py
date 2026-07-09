import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from dotenv import load_dotenv

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
current_dir    = os.path.dirname(os.path.abspath(__file__))
mimic_base_dir = os.path.normpath(os.path.join(current_dir, '..'))

data_dir    = os.path.join(mimic_base_dir, 'physionet.org', 'files', 'mimiciii', '1.4')
newdata_dir = os.path.join(mimic_base_dir, 'newdata')   # load.py 산출물 위치

print("="*20 + " [graph_builder.py] 이종 그래프 토폴로지 빌더 가동 " + "="*20)
print(f"➔ 메타 번들 경로: {newdata_dir}")

# ── STEP 1. 메타 번들 로드 ────────────────────────────────────────────────────
meta_path = os.path.join(newdata_dir, 'v5_processed_meta.pt')
if not os.path.exists(meta_path):
    raise FileNotFoundError(
        f"v5_processed_meta.pt 없음. load.py 를 먼저 실행하세요.\n경로: {meta_path}"
    )

print(">> [STEP 1] 메타 번들 로드 중...")
meta = torch.load(meta_path, weights_only=False)

valid_patients = meta["valid_patients"]
top_icd_codes  = meta["top_icd_codes"]
X_patient      = meta["X_patient"]   # [3000, 12]
X_disease      = meta["X_disease"]   # [50, 6]

patient_to_idx = {pid: idx for idx, pid in enumerate(valid_patients)}
icd_to_idx     = {code: idx for idx, code in enumerate(top_icd_codes)}

# ── STEP 2. 엣지 인덱싱 ──────────────────────────────────────────────────────
print(">> [STEP 2] DIAGNOSES_ICD 엣지 추출 중...")
diagnoses_path = os.path.join(data_dir, 'DIAGNOSES_ICD.csv.gz')
df_diag = pd.read_csv(diagnoses_path,
                      usecols=['SUBJECT_ID', 'ICD9_CODE'],
                      compression='gzip').dropna()

df_filtered = df_diag[
    (df_diag['SUBJECT_ID'].isin(patient_to_idx)) &
    (df_diag['ICD9_CODE'].isin(icd_to_idx))
]

edge_src, edge_dst = [], []
for _, row in df_filtered.iterrows():
    edge_src.append(patient_to_idx[int(row['SUBJECT_ID'])])
    edge_dst.append(icd_to_idx[row['ICD9_CODE']])

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

# ── STEP 3. HeteroData 구성 ───────────────────────────────────────────────────
print(">> [STEP 3] PyG HeteroData 객체 생성...")
data = HeteroData()

data['patient'].x          = X_patient
data['patient'].num_nodes  = len(valid_patients)
data['disease'].x          = X_disease
data['disease'].num_nodes  = len(top_icd_codes)

data['patient', 'has_disease', 'disease'].edge_index        = edge_index
data['disease', 'rev_has_disease', 'patient'].edge_index    = torch.stack(
    [edge_index[1], edge_index[0]], dim=0
)

# ── STEP 4. 저장 ─────────────────────────────────────────────────────────────
print(">> [STEP 4] hetero_graph.pt 저장 중...")
output_path = os.path.join(newdata_dir, 'hetero_graph.pt')
torch.save(data, output_path)

print("="*20 + " [완료] 그래프 저장 완료 " + "="*20)
print(f"➔ {output_path}")
