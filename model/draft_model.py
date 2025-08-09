import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from defined_functions import *
from data_processing import *
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data 

goa_path="../data/goa_human.gaf"
sequence_path='../data/train_sequences.tsv'
embedding_path='../data/model_vector/esm_swissprot_650U_500.pt'
embedding_path_owl2vec = '../data/model_vector/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'

### 数据预处理，找出所有包含有Annotation,且Annotation数量大于20的蛋白质
protein_ids,protein_embeddings,GO_list=load_filtered_protein_embeddings(goa_path,sequence_path, embedding_path)


mlp=EmbeddingTransform()
mlp=mlp.to('cuda' if torch.cuda.is_available() else 'cpu') ## Move model to GPU if available

with torch.no_grad():
    transformed_embeddings = mlp(protein_embeddings)

classes,model=load_owl2vec_embeddings(embedding_path_owl2vec,onto_path)
embedding_list=[]

adj,enc=create_adjacency_matrix(onto_path,GO_list,'biological_process')
edge_index,edge_attr= dense_to_sparse(adj)
edge_index=edge_index.to('cuda' if torch.cuda.is_available() else 'cpu')

label_num=len(enc.classes_)
for i in range(label_num):
    node=enc.inverse_transform([i])[0]
    embedding_list.append(torch.tensor(model.wv.get_vector("http://purl.obolibrary.org/obo/"+node)))
    
embedding_list=torch.stack(embedding_list)
embedding_vector=embedding_list.to('cuda' if torch.cuda.is_available() else 'cpu')

data=Data(x=embedding_vector,edge_index=edge_index).cuda()
GCN_model=GCN(input_dim=embedding_vector.shape[1], hidden_dim=512, output_dim=200).cuda()
with torch.no_grad():
    out = GCN_model(data.x, data.edge_index) 
##删除其中没有