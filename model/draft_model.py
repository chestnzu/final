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
embedding_path_owl2vec = '../data/pre_trained_model/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'

go_aspect=['biological_process', 'molecular_function', 'cellular_component']

### 数据预处理，找出所有包含有Annotation,且Annotation数量大于20的蛋白质

go_labels={'biological_process':[], 'molecular_function':[], 'cellular_component':[]}
print('sucessfully load the protein embeddings')

owl2vec_model=load_owl2vec_embeddings(embedding_path_owl2vec)
print('sucessfully load the OWL2VEC embeddings')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 200
num_layers = 6
num_heads = 8
epoch_num=30
e=math.e
metrics_output_test = {}
sigmoid=torch.nn.Sigmoid()
# protein_ids,protein_sequence,go_annotation_list,go_dict=load_filtered_protein_embeddings(goa_path,sequence_path)
## go list number may not be the same as label_num, as we are creating the adjacency matrix and all ancestors are included,
## some terms that are not in the go_list may be included in the adjacency matrix as they are ancestors of the terms in the go_list
for aspect in go_aspect:

    ## 1. load data
    train_data=pd.read_pickle('../data/dataset/{}/train_data.pkl'.format(aspect))
    valid_data=pd.read_pickle('../data/dataset/{}/valid_data.pkl'.format(aspect))
    test_data=pd.read_pickle('../data/dataset/{}/test_data.pkl'.format(aspect))

    ## 2. create edge_index for all annotated GO terms 
    all_data=pd.concat([train_data,valid_data,test_data],axis=0)
    annotated_go_term=list(set(chain.from_iterable(all_data.propagate_annotations)))
    edge_index,enc=create_edge_index(onto_path,aspect)
    edge_index=edge_index.to('cuda' if torch.cuda.is_available() else 'cpu')
    print('successfully create edge index for {}'.format(aspect))
    label_num=len(enc.classes_)

    ## 3. create graph data for GNN
    embedding_list=[]
    for i in range(label_num):
        node=enc.inverse_transform([i])[0]
        embedding_list.append(torch.tensor(owl2vec_model.wv.get_vector("http://purl.obolibrary.org/obo/"+node)))
    embedding_vector=torch.stack(embedding_list)
    embedding_vector=embedding_vector
    data=Data(x=embedding_vector,edge_index=edge_index).to(device)

    ## 4. create protein dataset
    train_dataset=build_dataset(train_dataset)
    val_dataset=build_dataset(valid_data)
    test_dataset=build_dataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    ## 5. model training
    combine_model=Combine_Transformer(input_dim=input_dim,output_dim=label_num,num_layers=num_layers,num_heads=num_heads,GO_data=data).to(device)
    loss_fn=nn.BCELoss() ##y 使用BCEWithLogitsLoss,不需要再使用 sigmoid
    optimizer = torch.optim.AdamW(combine_model.parameters(), lr=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)
    if aspect not in metrics_output_test:
        metrics_output_test[aspect] = {
                'f1_micro':[],
                'fmax':[],
                'aupr':[],
                'roc':[]
            }
        best_f1 = 0
        best_model_weights = None
        optimizer_model_weights = None

    for epoch in range(epoch_num):
        combine_model.train()
        loss_mean=0
        for i,batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            protein_ids = batch['protein_id']
            sequences = batch['sequence']
            golds = batch['labels'].cuda()
            protein_embeddings=load_protein_embeddings(protein_ids,embedding_path).cuda()
            protein_embeddings = protein_embeddings.to(device)
            output=combine_model(protein_embeddings)
            output=sigmoid(output)
            loss=loss_fn(output,golds)
            loss.backward()
            optimizer.step()
            loss_mean+=loss.item()
            if (i+1) % 10 == 0:
                print('{}  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(aspect, epoch + 1, epoch_num, i + 1,
                                                                                 (len(train_dataset) // 64)+1,
                                                                                 loss_mean / (i + 1)))    
        scheduler.step()
 ### ---- vliadation set ---- ###
        combine_model.eval()
        labels=[]
        preds=[]
        with torch.no_grad():
            for i,batch in tqdm(enumerate(val_dataloader)):
                protein_ids = batch['protein_id']
                sequences = batch['sequence']
                golds = batch['labels'].squeeze(0)
                protein_embeddings=load_protein_embeddings(protein_ids,embedding_path).cuda()
                protein_embeddings = protein_embeddings.to(device)
                output=combine_model(protein_embeddings).squeeze(0)
                labels.append(golds.cpu())
                preds.append(output.cpu())
        roc=cal_roc(preds,labels)
        fmax,_,_,_,_=evaluate_annotations(preds,labels)
        aupr=cal_aupr(preds,labels)
#        _, f1_micro,_,= cal_f1(preds,labels)
        # print('{}  Epoch: {}, Test F1-micro: {:.2f}%, Test Fmax:{:.2f}%, Test AUPR:{:.2f}%'.
        #         format(aspect, epoch + 1, 100 * f1_micro, 100 * fmax, 100 * aupr))
        print('{}  Epoch: {}, Test Fmax:{:.2f}%, Test AUPR:{:.2f}%'.
                format(aspect, epoch + 1, 100 * fmax, 100 * aupr))        
#        metrics_output_test[aspect]['f1_micro'].append(f1_micro)
        metrics_output_test[aspect]['fmax'].append(fmax)
        metrics_output_test[aspect]['aupr'].append(aupr)
        metrics_output_test[aspect]['roc'].append(roc)       
        f1 =fmax
        if f1 > best_f1:
            best_f1 = f1
            best_model_weights = combine_model.state_dict().copy()
#           optimizer_model_weights = optimizer.state_dict().copy()
            ckpt_path = '../data/model_checkpoint/'
            ckpt_path = ckpt_path + "{}_final_owl2vec_esm2_t30_650M_UR50D_{}.pt".format(ctime, aspect)
            torch.save(best_model_weights, ckpt_path)
    labels, preds = [], []
    combine_model.load_state_dict(torch.load(ckpt_path))
    with torch.no_grad():
        for batch in test_dataloader:
            protein_ids = batch['protein_id']
            golds = batch['labels'].cuda()
            protein_embeddings = load_protein_embeddings(protein_ids, embedding_path).to(device)

            output = combine_model(protein_embeddings)

            labels.append(golds.cpu())
            preds.append(output.cpu())

    roc=cal_roc(preds,labels)
    fmax=f_max(preds,labels)
    aupr=cal_aupr(preds,labels)
    # _, f1_micro,_,= cal_f1(preds,labels)
    print(f"F, Test Fmax: {100*fmax:.2f}%, Test AUPR: {100*aupr:.2f}%")