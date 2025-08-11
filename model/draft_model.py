import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from defined_functions import *
from data_processing import *
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data 
from torch.utils.data import DataLoader,Dataset,random_split
from tqdm import tqdm
import esm


goa_path="../data/goa_human.gaf"
sequence_path='../data/train_sequences.tsv'
embedding_path='../data/model_vector/esm_swissprot_650U_500.pt'
embedding_path_owl2vec = '../data/model_vector/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'

go_aspect=['biological_process', 'molecular_function', 'cellular_component']

### 数据预处理，找出所有包含有Annotation,且Annotation数量大于20的蛋白质
protein_ids,protein_sequence,go_annotation_list,go_list=load_filtered_protein_embeddings(goa_path,sequence_path)
go_labels={'biological_process':[], 'molecular_function':[], 'cellular_component':[]}
print('sucessfully load the protein embeddings')

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

for aspect in go_aspect:
    adj_matrix,enc,label_list=create_adjacency_matrix(onto_path,go_list,aspect)
    print('successfully create adjacency matrix for {}'.format(aspect))
    label_num=len(label_list)
    for x in go_annotation_list:
        x= x.split(';')
        temp_list=[term for term in x if term in label_list]
        if len(temp_list) == 0:
            go_labels[aspect].append([0]*label_num)
        else:
            digit_labels=enc.transform(temp_list)
            go_labels[aspect].append([1 if i in digit_labels else 0 for i in range(label_num)])
    datasets=zip(protein_sequence, protein_ids, go_labels[aspect])
    train_size= int(0.8 * len(protein_ids))
    val_size = int(0.1 * len(protein_ids))
    test_size = len(protein_ids) - train_size - val_size
    datasets= protein_loader(datasets)
    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #optimizer = torch.optim.Adam(model_mlp.parameters(), lr=4e-5)  # 4e-5 150M 1e-5 3B
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)

    for batch in tqdm(train_dataloader):
        protein_ids = batch['protein_id']
        sequences = batch['sequence']
        annotations = batch['labels']
        protein_embeddings=load_protein_embeddings(sequences, protein_ids, model, batch_converter, alphabet).cuda()
        

    

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
    output = GCN_model(data.x, data.edge_index) 

