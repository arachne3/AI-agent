import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import negative_sampling
from dotenv import load_dotenv

load_dotenv()

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
current_dir    = os.path.dirname(os.path.abspath(__file__))
mimic_base_dir = os.path.normpath(os.path.join(current_dir, '..'))
newdata_dir    = os.path.join(mimic_base_dir, 'newdata')   # graph_builder 산출물 위치

print("="*20 + " [model.py] ClinicalHeteroGNN 학습 엔진 가동 " + "="*20)
print(f"➔ 그래프 로드 경로: {newdata_dir}")


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class ClinicalHeteroGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_norm = nn.BatchNorm1d(12)
        self.d_norm = nn.BatchNorm1d(6)

        self.conv1 = HeteroConv({
            ('patient', 'has_disease', 'disease'):     SAGEConv((12, 6),   64, aggr='mean'),
            ('disease', 'rev_has_disease', 'patient'): SAGEConv((6,  12),  64, aggr='mean'),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('patient', 'has_disease', 'disease'):     SAGEConv((64, 64), 128, aggr='mean'),
            ('disease', 'rev_has_disease', 'patient'): SAGEConv((64, 64), 128, aggr='mean'),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        x_scaled = {
            'patient': self.p_norm(x_dict['patient']),
            'disease': self.d_norm(x_dict['disease']),
        }
        out = self.conv1(x_scaled, edge_index_dict)
        out = {k: F.relu(v) for k, v in out.items()}
        out = self.conv2(out, edge_index_dict)
        return out


class CosineLinkPredictor(nn.Module):
    def forward(self, h_patient, h_disease, edge_index):
        src = F.normalize(h_patient[edge_index[0]], p=2, dim=-1)
        dst = F.normalize(h_disease[edge_index[1]], p=2, dim=-1)
        return torch.sum(src * dst, dim=-1) * 5.0


# ── 학습 및 임베딩 추출 ───────────────────────────────────────────────────────
def train_and_extract():
    graph_path = os.path.join(newdata_dir, 'hetero_graph.pt')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"hetero_graph.pt 없음. graph_builder.py 를 먼저 실행하세요.\n경로: {graph_path}"
        )

    data   = torch.load(graph_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"➔ 연산 장치: {device}")

    model     = ClinicalHeteroGNN().to(device)
    predictor = CosineLinkPredictor().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.001, weight_decay=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    pos_edge_index = data['patient', 'has_disease', 'disease'].edge_index.to(device)
    x_dict         = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

    num_patients = data['patient'].num_nodes
    num_diseases = data['disease'].num_nodes

    print(">> [TRAINING] 150 에포크 학습 시작...")
    model.train()

    for epoch in range(1, 151):
        optimizer.zero_grad()

        h_dict    = model(x_dict, edge_index_dict)
        h_patient = h_dict['patient']
        h_disease = h_dict['disease']

        pos_out    = predictor(h_patient, h_disease, pos_edge_index)
        pos_labels = torch.ones(pos_out.size(0), device=device)

        neg_edge   = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(num_patients, num_diseases),
            num_neg_samples=pos_edge_index.size(1),
        ).to(device)
        neg_out    = predictor(h_patient, h_disease, neg_edge)
        neg_labels = torch.zeros(neg_out.size(0), device=device)

        loss = criterion(
            torch.cat([pos_out, neg_out]),
            torch.cat([pos_labels, neg_labels])
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch {epoch:03d}/150 | Loss: {loss.item():.4f}")

    print(">> [EXTRACT] 임베딩 추출 및 저장 중...")
    model.eval()
    with torch.no_grad():
        final = model(x_dict, edge_index_dict)
        z_patient = final['patient'].cpu()
        z_disease = final['disease'].cpu()

    torch.save(z_patient, os.path.join(newdata_dir, 'z_patient.pt'))
    torch.save(z_disease, os.path.join(newdata_dir, 'z_disease.pt'))

    print("="*20 + " [완료] 임베딩 저장 완료 " + "="*20)
    print(f"➔ z_patient: {list(z_patient.shape)}")
    print(f"➔ z_disease: {list(z_disease.shape)}")
    print(f"➔ 저장 경로: {newdata_dir}")


if __name__ == "__main__":
    train_and_extract()
