import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import obonet
from owlready2 import get_ontology



class EmbeddingTransform(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=200):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
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

def rdf_to_edge_index(onto_path,namespace):
    g = nx.DiGraph()
    onto=get_ontology(onto_path).load()
    node_list = []
    for cls in onto.classes():
        if len(cls.hasOBONamespace)==0:
            continue
        if len(str(cls)) > 14 or str(cls)=='owl.Thing' or cls.hasOBONamespace[0]!=namespace:  # Filter out classes with long names
            continue
        for parent in cls.is_a:
            if str(parent) != 'owl.Thing' and len(str(parent))<=14:  # Avoid adding the root class
                g.add_edge(str(parent), str(cls), label='subClassOf')
                if parent.iri not in node_list:
                    node_list.append(parent.iri)
                if cls.iri not in node_list:
                    node_list.append(cls.iri)
    idx2node = {i: node for i, node in enumerate(node_list)}   
    for node in g.nodes:
        g.nodes[node]['label'] = str(node)
    pygdata=from_networkx(g)

    return pygdata,node_list,idx2node