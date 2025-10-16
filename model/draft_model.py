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
import math
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from itertools import chain 


ctime = datetime.now().strftime("%Y%m%d%H%M%S")
goa_path="../data/goa_human.gaf"
sequence_path='../data/esm2650M_swissprot_human.pt'
embedding_path='../data/esm2650M_swissprot_human.pt'
embedding_path_owl2vec = '../data/pre_trained_model/ontology.embeddings'
onto_path='../data/go.obo'

go_aspect=['bp', 'mf', 'cc']
model_file='../data/model_checkpoint/20231011143021_best_owl2vec_esm2_t30_650M_UR50D_bp.pt'
owl2vec_model=load_owl2vec_embeddings(embedding_path_owl2vec)
print('sucessfully load the OWL2VEC embeddings')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 200
output_dim = 32
num_layers = 6
num_heads = 8
epoch_num=30
hidden_dim=64
e=math.e
## load protein embeddings
protein_labels,_,protein_embeddings=torch.load(embedding_path)


## go list number may not be the same as label_num, as we are creating the adjacency matrix and all ancestors are included,
## some terms that are not in the go_list may be included in the adjacency matrix as they are ancestors of the terms in the go_list
for aspect in go_aspect:

    ## 1. load data
    train_data=pd.read_pickle('../data/dataset/{}/train_data.pkl'.format(aspect))
    valid_data=pd.read_pickle('../data/dataset/{}/valid_data.pkl'.format(aspect))
    test_data=pd.read_pickle('../data/dataset/{}/test_data.pkl'.format(aspect))

    ## 2. create edge_index for all annotated GO terms 
    all_data=pd.concat([train_data,valid_data,test_data],axis=0)
    annotated_go_term=list(set(chain.from_iterable(all_data.propagate_annotation)))
    edge_index,enc=create_edge_index(onto_path,annotated_go_term,aspect)
    edge_index=edge_index.to(device)
    print('successfully create edge index for {}'.format(aspect))
    label_num=len(enc.classes_)

    ## 3. create graph data for GNN
    embedding_list=[]
    for i in range(label_num):
        node=enc.inverse_transform([i])[0]
        embedding_list.append(torch.tensor(owl2vec_model.wv.get_vector("http://purl.obolibrary.org/obo/"+node.replace(':','_'))))
    embedding_vector=torch.stack(embedding_list).to(device)
    data=Data(x=embedding_vector,edge_index=edge_index).to(device)

    ## 4. create protein dataset
    train_dataset=build_dataset(train_data,enc)
    val_dataset=build_dataset(valid_data,enc)
    test_dataset=build_dataset(test_data,enc)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    ## 5. model training
    combine_model=Combine_Transformer(input_dim=input_dim,num_layers=num_layers,num_heads=num_heads,GO_data=data,output_dim=output_dim,hidden_dim=hidden_dim).to(device)
    loss_fn=nn.BCELoss() 
    optimizer = torch.optim.AdamW(combine_model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)


## -- training -- ##
    best_loss=100000.00
    for epoch in range(epoch_num):
        combine_model.train()
        loss_mean=0
        for i,batch in tqdm(enumerate(train_dataloader)):
            protein_ids = batch['protein_id']
            train_labels = batch['labels'].to(device)
            protein_embeddings=load_protein_embeddings(protein_ids,protein_embeddings,protein_labels).to(device)
            train_output=combine_model(protein_embeddings)
            loss=loss_fn(train_output,train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mean+=loss.detach().item()
            if (i+1) % 10 == 0:
                print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(aspect, epoch + 1, epoch_num, i + 1,
                                                                                 (len(train_dataset) // 64)+1,
                                                                                 loss_mean / (i + 1)))    

 ### ---- validation set ---- ###
        combine_model.eval()
        labels, preds = [], []
        valid_mean_loss=0
        with torch.no_grad():

            for i,batch in tqdm(enumerate(valid_dataloader)):
                protein_ids = batch['protein_id']
                valid_labels = batch['labels'].to(device)
                protein_embeddings=load_protein_embeddings(protein_ids,protein_embeddings,protein_labels).to(device)
                protein_embeddings = protein_embeddings.to(device)
                valid_output=combine_model(protein_embeddings)
                valid_loss=loss_fn(valid_output,valid_labels)
                valid_mean_loss+=valid_loss.detach().item()
                labels.append(valid_labels.cpu())
                preds.append(valid_output.cpu())
        valid_loss=valid_mean_loss/(i+1)
        roc=cal_roc(preds,labels)
        print('AUC: {}'.format(roc))
        if valid_loss<best_loss:
            best_loss=valid_loss
            print('New record of loss on validation set: {:.4f}'.format(best_loss))
            torch.save(combine_model.state_dict(), model_file)
        scheduler.step()          

    
    combine_model.load_state_dict(torch.load(model_file))
    combine_model.eval()
    labels, preds = [], []
    with torch.no_grad():
        test_loss_mean=0
        for i,batch in tqdm(enumerate(test_dataloader)):
            protein_ids = batch['protein_id']
            test_labels = batch['labels'].to(device)
            protein_embeddings = load_protein_embeddings(protein_ids,protein_embeddings,protein_labels).to(device)
            test_output = combine_model(protein_embeddings)
            test_loss = loss_fn(test_output, test_labels)
            test_loss_mean += test_loss.detach().item()
            labels.append(test_labels.cpu())
            preds.append(test_output.cpu())
        test_roc=cal_roc(preds,labels)
        print('Loss: {:.4f},AUC: {}'.format((test_loss_mean / (i + 1)),test_roc))    
    with open(f'../data/model_evaluation/{aspect}/model.txt', 'w') as f:
        f.write(f'Loss: {test_loss_mean / (i + 1):.4f}, AUC: {test_roc}\n')