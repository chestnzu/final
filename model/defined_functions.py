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
from skelearn.metrics import f1_score,roc_auc_score



class EmbeddingTransform(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, output_dim=200):
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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=0.8)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.pool2 = TopKPooling(output_dim, ratio=0.8)

    def forward(self, x, edge_index):
        batch = torch.zeros(x.size(0), dtype=torch.long).cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index,batch=batch)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index,batch=batch)
        go_pooled = x.mean(dim=0, keepdim=True)  # Global pooling
        return go_pooled




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
    def __init__(self, input_dim, output_dim,num_layers,num_heads,GO_data):
        super(Combine_Transformer, self).__init__()
        self.heads = num_heads
        self.go_data = GO_data
        self.pool1= TopKPooling(input_dim, ratio=0.8)   
        self.fc1 = EmbeddingTransform()
        self.fc2 = nn.Linear(input_dim*2, output_dim)
        self.GCN = GCN(GO_data.x.shape[1], 16, input_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim*2, num_heads=num_heads)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim*2, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    
    def forward(self, protein_vectors):
        protein_vectors = self.fc1(protein_vectors) ## 将ESM2的1280维向量转换为200维向量
        go_vectors= self.GCN(self.go_data.x, self.go_data.edge_index) ## 将OWL2VEC的200维向量作为输入，放入GCN中，并加上GO的邻接矩阵，进一步训练，生成对应的200维向量
        go_expand= go_vectors.expand(protein_vectors.size(0),-1)  # 扩展维度以匹配蛋白质向量的批次大小
        combine_features = torch.cat((protein_vectors, go_expand), dim=1)
        attn_output, _ = self.multihead_attn(combine_features, combine_features, combine_features)
        transformer_output = self.transformer_encoder(attn_output)
        output = self.fc2(transformer_output)
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

def load_protein_embeddings(sequences,protein_ids,model,batch_converter,alphabet):
    batch_input = [(protein_id, seq) for protein_id, seq in zip(protein_ids, sequences)]
    sequence_representations = []
    for i in range(0,len(protein_ids),2):
        micro_batch = batch_input[i:i+2]
        batch_labels,batch_strs,batch_tokens = batch_converter(micro_batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33].detach().cpu()
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    embedding_batch= torch.stack(sequence_representations, dim=0)
    return embedding_batch

def cal_f1(preds, golds):
    preds[preds>=0.5]=1
    preds[preds<0.5]=0
    f1_macro = f1_score(golds, preds, average='macro')
    f1_micro = f1_score(golds, preds, average='micro')
    f1_sample = f1_score(golds, preds, average='samples')

    return f1_macro, f1_micro, f1_sample

def cal_roc(preds, golds):
    auc_micro = roc_auc_score(golds, preds, average="samples")  
    return auc_micro

## TODO ##
## 1. add AUPR
## 2. add F-max