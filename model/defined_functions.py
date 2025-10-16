import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.utils import to_dense_adj
import networkx as nx
import obonet,math
from owlready2 import get_ontology
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import esm
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve,average_precision_score,roc_curve,auc
import numpy as np
import obonet
from dataset_generating.basics import Ontology


class EmbeddingTransform(nn.Module):
    def __init__(self, input_dim=1280, output_dim=200):
        super().__init__()       
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        self.norm(x)
        self.dropout(x)
        return x


class Attention_layer(nn.Module):
    def  __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels),
        )
    def forward(self, x):
        ### 输入：batch * GO_term_num * 200
        ### 输出：batch * GO_term_num * 200
        if x.dtype == torch.float32:
            x = x.float()
        Z = self.model(x) ## batch * GO_term_num * 200
        # score = torch.bmm(Z, Z.transpose(1, 2))  # batch * GO_term_num * GO_term_num
        score = torch.sparse.mm(Z, Z.t())  # batch * GO_term_num * GO_term_num
        W = F.softmax(score, dim=1).to(torch.float32) # 归一化为0-1权重
        return W

class ProteinGoCrossAttention(nn.Module):
    def __init__(self, protein_dim, go_dim, num_go, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_go, hidden_dim))  # learnable GO queries
        self.key = nn.Linear(protein_dim, hidden_dim)
        self.value = nn.Linear(protein_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, go_dim)
        self.go_dim = go_dim

    def forward(self, protein_emb):
        # protein_emb: (batch, protein_dim)
        K = self.key(protein_emb).unsqueeze(1)   # (batch, 1, hidden_dim)
        V = self.value(protein_emb).unsqueeze(1) # (batch, 1, hidden_dim)
        Q = self.query.unsqueeze(0).expand(protein_emb.size(0), -1, -1)  # (batch, num_go, hidden_dim)
        
        attention = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) / (self.go_dim ** 0.5), dim=-1)
        out = torch.matmul(attention, V)  # (batch, num_go, hidden_dim)
        output = self.out(out)               # (batch, num_go, go_dim)
        return output

class GoAttentionPooling(nn.Module):
    def __init__(self, num_go_reduced, go_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_go_reduced, go_dim))
        self.key_proj = nn.Linear(go_dim, go_dim)
        self.value_proj = nn.Linear(go_dim, go_dim)
        self.go_dim = go_dim  # 保存为实例属性
        
    def forward(self, go_matrix):
        # go_matrix: (batch, num_go_full, go_dim)
        K = self.key_proj(go_matrix)       # (batch, num_go_full, go_dim)
        V = self.value_proj(go_matrix)     # (batch, num_go_full, go_dim)
        Q = self.query.unsqueeze(0).expand(go_matrix.size(0), -1, -1)  # (batch, num_go_reduced, go_dim)
        
        attention = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) / (self.go_dim ** 0.5), dim=-1)
        out = torch.matmul(attention, V)   # (batch, num_go_reduced, go_dim)
        return out

class cheb_conv_K(nn.Module):
    def __init__(self, K, in_channels, out_channels, device):
        super(cheb_conv_K, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels).to(self.DEVICE)).to(torch.float32)

    def forward(self, x, adj):
        '''
        Chebyshev graph convolution operation
        '''
        graph_signal = x  # (class, F_in)
        T_k_with_at = adj  # (class, class)
        output = torch.zeros_like(graph_signal).to(self.DEVICE)  #  class, F_out)
        for k in range(self.K):
            theta_k = self.Theta[k]  # (in_channels, out_channels)
            rhs = torch.sparse.mm(T_k_with_at, graph_signal)  # (class, F_in) * (class, class) -> (class, F_in)
            output += torch.matmul(rhs, theta_k)  # (class, F_in) * (F_in, F_out) -> (class, F_out)

        return F.relu(output)  # (class, F_out)

class protein_loader(Dataset):
    def __init__(self, dataset):
        sequences,protein_ids,exp_annotations,annotations= zip(*dataset)
        self.sequences = sequences
        self.protein_ids = protein_ids
        self.annotations = annotations ## one-hot vector, encoding which GO terms are annotated to the protein
        self.exp_annotations = exp_annotations

    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[idx]
        annotations = self.annotations[idx]
        exp_annotations = self.exp_annotations[idx]
        return {'protein_id': protein_id, 'sequence':sequence, 'exp_labels':torch.as_tensor(exp_annotations,dtype=torch.float32).clone().detach(),'labels': torch.as_tensor(annotations,dtype=torch.float32).clone().detach()}
    
class Combine_Transformer(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, GO_data,output_dim,hidden_dim):
        super(Combine_Transformer, self).__init__()
        self.heads = num_heads
        self.go_data = GO_data
        self.fc1 = EmbeddingTransform(output_dim=input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, 1)
        self.fc4 = nn.Linear(self.go_data.x.shape[0]//8, self.go_data.x.shape[0])
        self.GCN = cheb_conv_K(K=3, in_channels=input_dim, out_channels=input_dim, device='cuda')
        self.convert_go_feature = ProteinGoCrossAttention(protein_dim=input_dim, num_go=GO_data.x.shape[0]//8, go_dim=output_dim,hidden_dim=hidden_dim) #protein_dim, go_dim, num_go, hidden_dim
        self.compressGO = GoAttentionPooling(num_go_reduced=GO_data.x.shape[0]//8, go_dim=output_dim)
        self.training = True
        
        # 优化的norm配置
        self.norm1 = nn.BatchNorm1d(input_dim).to(torch.float32)  # GCN后：图数据用BatchNorm
        self.norm2 = nn.LayerNorm(output_dim).to(torch.float32)    # 融合特征后：稳定注意力输入
        self.norm3 = nn.LayerNorm(output_dim).to(torch.float32)
        # norm3 可以省略，因为TransformerEncoder内部有LayerNorm
        self.functionAttention = Attention_layer(in_channels=input_dim, hid_channels=256, out_channels=input_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, 
                                                dropout=0.1, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads, 
                                                           dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)     
        self.dropout = nn.Dropout(0.1)
        
   
    def forward(self, protein_vectors):
        ## 1. 蛋白质特征变换
        ## 输入特征为 batch * 1280, 输出为 batch * 200
        protein_vectors = self.fc1(protein_vectors)
        protein_vectors_norm = self.norm1(protein_vectors)
        ## 2. 将蛋白质特征转换为GO维度的特征
        protein_go_features = self.convert_go_feature(protein_vectors_norm)  # batch * GO_term_num * 200
        ## 3. GO特征图卷积 + BatchNorm + Dropout
        # go_features = self.go_data.x.unsqueeze(0).expand(protein_vectors.shape[0],-1,-1)  # batch * GO_term_num * 200
        go_features = self.go_data.x
        go_adj = to_dense_adj(self.go_data.edge_index)[0]
        attention_adj = self.functionAttention(go_features)  ## batch * GO_term_num * GO_term_num        
        go_adj = go_adj + attention_adj ## batch * GO_term_num * GO_term_num
        go_output = self.GCN(go_features,go_adj) +go_features ## batch * GO_term_num * 200
        go_output = go_output.unsqueeze(0).expand(protein_vectors.shape[0],-1,-1) ## batch * GO_term_num * 200
        go_output = self.fc2(go_output)  # batch * GO_term_num * output_dim
        go_output = self.norm2(go_output)
        go_features = F.dropout(go_output, p=0.1, training=self.training) # batch * GO_term_num * output_dim
        go_features = self.compressGO(go_features)  # batch * (GO_term_num//8) * output_dim

        ## 4. 特征融合
        combine_features = protein_go_features + go_features  # batch * (GO_term_num//8) * output_dim
        # combine_features = self.fc2(combine_features)
        # combine_features = self.dropout(combine_features)
        combine_features = self.norm3(combine_features)
        attn_output, _ = self.multihead_attn(combine_features, combine_features, combine_features)
        
        # 5. Transformer编码器（内部已有LayerNorm和残差连接）
        transformer_output = self.transformer_encoder(attn_output) ## batch * (GO_term_num//8) * output_dim
        
        # 6. 最终输出
        output = self.fc3(transformer_output).squeeze(-1)  # batch * GO_term_num//8
        output = self.fc4(output)  # batch * GO_term_num
        output == torch.sigmoid(output) # batch * GO_term_num
        return output



def create_edge_index(onto_path,go_list,aspect):
    enc=LabelEncoder()
    onto = Ontology(onto_path, with_rels=True)
    aspect_map={'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    namespace_terms=onto.get_namespace_terms(aspect_map[aspect])
    go_list = list(set(go_list).intersection(namespace_terms))
    enc.fit(go_list)
    mapping={go_id: idx for idx, go_id in enumerate(enc.classes_)}
    nx_onto=obonet.read_obo(onto_path)
    onto_1=nx.relabel_nodes(nx_onto, mapping)
    go_list_digit=[idx for idx, _ in enumerate(enc.classes_)]
    edges=[(a,b) for a,b in onto_1.edges() if a in go_list_digit and b in go_list_digit]
    src,dst = zip(*edges)
    edge_index=torch.tensor([src,dst],dtype=torch.long)
    return edge_index, enc


def load_protein_embeddings(protein_ids, embedding,label):
    sequence_representations = []
    for pid in protein_ids:
        if pid in label:
            idx = label.index(pid)
            sequence_representations.append(embedding[idx])
        else:
            sequence_representations.append(torch.zeros(1280))
    embedding_batch = torch.stack(sequence_representations)
    return embedding_batch

def cal_f1(preds, golds):
    f1_macro = f1_micro = f1_sample = 0
    total = len(preds)

    for i in range(total):
        _preds = np.array(preds[i].cpu())
        _golds = np.array(golds[i].cpu())
        _preds[_preds >= 0.5] = 1
        _preds[_preds < 0.5] = 0

        f1_macro += f1_score(_golds, _preds, average='macro', zero_division=1)
        f1_micro += f1_score(_golds, _preds, average='micro', zero_division=1)
        f1_sample += f1_score(_golds, _preds, average='samples', zero_division=1)

    return f1_macro/total, f1_micro/total, f1_sample/total

def cal_roc(preds, golds):

    _preds = np.array(preds).flatten()
    _golds = np.array(golds).flatten()
    fpr,tpr,_ = roc_curve(_golds, _preds)
    roc_auc=auc(fpr,tpr)

    return roc_auc

def f_max(preds,golds):
    f_max=0
    for i in range(len(preds)):
        _preds = np.array(preds[i].cpu())
        _golds = np.array(golds[i].cpu())
        valid_labels = [j for j in range(_golds.shape[1]) if len(np.unique(_golds[:, j])) > 1]
        if not valid_labels:  # 没有合法标签
            return 0
        _preds = _preds[:, valid_labels]
        _golds = _golds[:, valid_labels]
        precision, recall, thresholds = precision_recall_curve(_golds.ravel(),_preds.ravel())
        f_scores = 2 * precision * recall / (precision + recall + 1e-8)
        if f_scores.max() > f_max:
            f_max = f_scores.max()
            best_threshold = thresholds[f_scores.argmax()]
    return f_max

def cal_aupr(preds, golds):
    _preds = np.concatenate([np.array(p.cpu()) for p in preds], axis=0)
    _golds = np.concatenate([np.array(g.cpu()) for g in golds], axis=0)

    valid_labels = [j for j in range(_golds.shape[1]) if len(np.unique(_golds[:, j])) > 1]
    if not valid_labels:
        return 0
    _preds = _preds[:, valid_labels]
    _golds = _golds[:, valid_labels]
    aupr_micro = average_precision_score(_golds, _preds, average="micro")

    return aupr_micro

def evaluate_annotations(real_annots, pred_annots):
    """
    Computes Fmax, Smin, WFmax and Average IC
    Args:
       real_annots (set): Set of real GO classes
       pred_annots (set): Set of predicted GO classes
    """
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = set(pred_annots[i]) - tp
        fn = set(real_annots[i]) - tp
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn)) if (tpn + fnn) > 0 else 0
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn)) if (tpn + fpn) > 0 else 0
            p += precision

    r /= total if total > 0 else 0
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    return f, p, r, fps, fns


def build_dataset(dataset,enc):
    protein_name=list(dataset['proteins'].values)
    sequence=list(dataset['sequences'].values)
    num_terms = len(enc.classes_)  # one-hot 向量长度（GO term 总数）

    exp_labels = []
    labels = []
    for go_list in dataset['exp_annotations']:
        vec = np.zeros(num_terms, dtype=np.float32)
        for go in go_list:
            if go in enc.classes_:
                vec[np.where(enc.classes_ == go)[0][0]] = 1.0
        exp_labels.append(vec)

    for go_list in dataset['propagate_annotation']:
        vec = np.zeros(num_terms, dtype=np.float32)
        for go in go_list:
            if go in enc.classes_:
                vec[np.where(enc.classes_ == go)[0][0]] = 1.0
        labels.append(vec)
    dataset=zip(sequence,protein_name,exp_labels,labels)
    dataset=protein_loader(dataset)
    return dataset