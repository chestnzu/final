import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling,SAGEConv
from torch_geometric.utils import from_networkx
import networkx as nx
import obonet,math
from owlready2 import get_ontology
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import esm
from sklearn.metrics import f1_score,roc_auc_score,precision_recall_curve,average_precision_score
import numpy as np



class EmbeddingTransform(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=200):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.kaiming_normal_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
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




class protein_loader(Dataset):
    def __init__(self, dataset):
        sequences,protein_ids,annotations= zip(*dataset)
        self.sequences = sequences
        self.protein_ids = protein_ids
        self.annotations = annotations ## one-hot vector, encoding which GO terms are annotated to the protein

    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        sequence = self.sequences[idx]
        annotations = self.annotations[idx]
        return {'protein_id': protein_id, 'sequence':sequence, 'labels': torch.as_tensor(annotations, dtype=torch.float32).clone().detach()}
    
class Combine_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_heads, GO_data):
        super(Combine_Transformer, self).__init__()
        self.heads = num_heads
        self.go_data = GO_data
        self.fc1 = EmbeddingTransform()
        self.fc2 = nn.Linear(self.go_data.x.shape[0], input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
        self.GCN = GNN(GO_data.x.shape[1], 256, input_dim)
        
        # 优化的norm配置
        self.norm1 = nn.BatchNorm1d(input_dim).to(torch.float32)  # GCN后：图数据用BatchNorm
        self.norm2 = nn.LayerNorm(input_dim).to(torch.float32)    # 融合特征后：稳定注意力输入
        # norm3 可以省略，因为TransformerEncoder内部有LayerNorm
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, 
                                                   dropout=0.1, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, 
                                                           dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.1)
   
    def forward(self, protein_vectors):
        # 1. 蛋白质特征变换
        protein_vectors = self.fc1(protein_vectors)
        
        # 2. GO特征提取 + BatchNorm（图卷积后用BatchNorm）
        go_vectors = self.GCN(self.go_data.x, self.go_data.edge_index)
        go_vectors = self.norm1(go_vectors)
        go_vectors = F.relu(go_vectors)  # 激活函数
        
        # 3. 特征融合
        combine_features = protein_vectors @ torch.transpose(go_vectors, 0, 1)
        combine_features = self.fc2(combine_features)
        combine_features = self.dropout(combine_features)
        
        # 4. LayerNorm + 多头注意力（序列数据用LayerNorm）
        combine_features = self.norm2(combine_features)
        attn_output, _ = self.multihead_attn(combine_features, combine_features, combine_features)
        
        # 5. Transformer编码器（内部已有LayerNorm和残差连接）
        transformer_output = self.transformer_encoder(attn_output)
        
        # 6. 最终输出
        output = self.fc3(transformer_output)
        output = torch.sigmoid(output)
        return output



def create_adjacency_matrix(onto_path,go_list,namespace):
    enc=LabelEncoder()
    onto=get_ontology(onto_path).load()
    label_list=[]
    for cls in go_list:
        cls=onto.search_one(iri=cls.replace('GO_','http://purl.obolibrary.org/obo/GO_'))
        if cls.hasOBONamespace and cls.hasOBONamespace[0] == namespace:
            ancestors = cls.ancestors()
            if len(ancestors) > 0:
                label_list.extend(x.name for x in ancestors)
                label_list.append(cls.name)
                label_list= list(set(label_list))
        else:
            continue
    label_list = [x for x in label_list if x[:2]=='GO']  # Filter out terms with length > 14
    label_space=enc.fit_transform(label_list)
    label_num=len(label_space)
    print(label_num)
    adj_matrix=torch.zeros((label_num,label_num)).cuda()
    valid_list=list(set(go_list) & set(label_list))
    for term in valid_list:
        term=onto.search_one(iri=term.replace('GO_','http://purl.obolibrary.org/obo/GO_'))
        parents=term.is_a
        idx=enc.transform([term.name])
        if len(parents) == 0:
            continue
        for parent in parents:
            if str(parent) != 'owl.Thing' and len(str(parent))<=14:
                parent_idx = enc.transform([parent.name])
                adj_matrix[idx, parent_idx] = 1
    return adj_matrix, enc,label_list

def load_protein_embeddings(protein_ids, embedding_path):
    sequence_representations = []
    label,_,embedding=torch.load(embedding_path)
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
    # 拼接所有 batch
    _preds = np.concatenate([np.array(p.cpu()) for p in preds], axis=0)
    _golds = np.concatenate([np.array(g.cpu()) for g in golds], axis=0)
    valid_labels = [j for j in range(_golds.shape[1]) if len(np.unique(_golds[:, j])) > 1]
    if not valid_labels:  # 没有合法标签
        return 0
    _preds = _preds[:, valid_labels]
    _golds = _golds[:, valid_labels]
    try:
        #auc_macro = roc_auc_score(_golds, _preds, average="macro")
        auc_micro = roc_auc_score(_golds, _preds, average="micro")
    except ValueError:
        # 如果所有标签都是0，roc_auc_score会报错
        return 0

    return auc_micro

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
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
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