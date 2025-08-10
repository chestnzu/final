import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import obonet,math
from owlready2 import get_ontology
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset



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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class protein_loader(Dataset):
    def __init__(self, sequences, protein_ids,annotations):
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
    def __init__(self, input_dim, output_dim,num_layers,num_heads):
        super(Combine_Transformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    
    def forward(self, protein_vectors, go_vectors):
        combine_features = torch.cat((protein_vectors, go_vectors), dim=0)
        attn_output, _ = self.multihead_attn(combine_features, combine_features, combine_features)
        transformer_output = self.transformer_encoder(attn_output)
        output = self.fc(transformer_output)
        return output




def create_adjacency_matrix(onto_path,namespace):
    enc=LabelEncoder()
    onto=get_ontology(onto_path).load()
    label_list=[]
    for cls in onto.classes():
        if cls.hasOBONamespace and cls.hasOBONamespace[0] == namespace:
            label_list.append(cls.name)
        else:
            continue
    label_space=enc.fit_transform(label_list)
    label_num=len(label_space)
    adj_matrix=torch.zeros((label_num,label_num)).cuda()
    for term in label_list:
        term=onto.search_one(iri=term.replace('GO_','http://purl.obolibrary.org/obo/GO_'))
        parents=term.is_a
        idx=enc.transform([term.name])
        if len(parents) == 0:
            continue
        for parent in parents:
            if str(parent) != 'owl.Thing' and len(str(parent))<=14 and parent.name in label_list:
                parent_idx = enc.transform([parent.name])
                adj_matrix[idx, parent_idx] = 1
    return adj_matrix, enc,label_list