import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from all_models import *
from data_processing import *
from torch.utils.data import DataLoader
from defined_functions import *
import math
from datetime import datetime
from torch.nn import functional as F
import argparse
import json
from evaluation import *
from tqdm import tqdm
from comparing_models import *
from torch.optim.lr_scheduler import MultiStepLR
import click as ck
from multiprocessing import Pool
from functools import partial
from dataset_generating.basics import propagate_annots


@ck.command()
@ck.option('--aspect','-asp',default='mf',type=ck.Choice(['mf','cc','bp']),
           help='GO aspect')
@ck.option('--deepgo2','-dp',is_flag=True,help='if deepgo2 model is used')
@ck.option('--fastdataloader','-fdl',is_flag=True,help='if use fast dataloader from deepgo2')
@ck.option('--protein_embedding_path','-ep',default='../data/esm_embeddings_3B_complete.pt')
@ck.option('--go_embedding_path','-ge',default='../data/go_all_embeddings.pt')


def main(aspect,deepgo2,fastdataloader,protein_embedding_path,go_embedding_path):
    if deepgo2:
        onto_path = '../../deepgo2/data/go.obo'
        data_root = '../../deepgo2/data'
    else:
        onto_path = '../data/go.obo'
        data_root = '../data/dataset/'

    with open("./config/model_config.json", "r") as f:
        _cfg = json.load(f)
        config = _cfg['profiles']
    profile = config['default']
    epoch_num = profile['epoch_num']
    TRAIN_BS = profile['train_bs']
    EVAL_BS = profile['eval_bs']
    num_heads = profile['num_heads']
    num_layers = profile['num_layers']
    hidden_dim = profile['hidden_dim']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ## load protein embeddings##
    if deepgo2:
        protein_labels,protein_embeddings_all = load_deepgo2_data(data_root,'mf')
    else:
        embedding_data=torch.load(protein_embedding_path)
        protein_labels,_,protein_embeddings_all=embedding_data.values()
    protein_embeddings_all=protein_embeddings_all.to(device)
    ## dimension of protein features ##
    ## load context embedding ##
    go_data=torch.load(go_embedding_path,weights_only=False)
    go_list = go_data["terms"]
    go_context_embeddings = go_data["context_embeddings"]
    go_embeddings = go_data["go_embeddings"]

    ## load ontology ##
    go=Ontology(onto_path,with_rels=True)
    model_save_file_path=f'../data/model_checkpoint/best_{aspect}.pt'
    train_df,valid_df,test_df,terms,terms_dict,termidx = load_data(aspect,data_root)
    term_idxes=[idx for term,idx in terms_dict.items() if term in go_list]

    ## only keep go information that are annotated in the training set ##
    go_context_embeddings=torch.from_numpy(go_context_embeddings[term_idxes]).float().to(device)
    go_embeddings=torch.from_numpy(go_embeddings[term_idxes]).float().to(device)

    ## build dataloader ##
    train_dataset=build_dataset(train_df,terms_dict,protein_embeddings_all,protein_labels,fdl=fastdataloader)  ###
    val_dataset=build_dataset(valid_df,terms_dict,protein_embeddings_all,protein_labels,fdl=fastdataloader)  ###
    test_dataset=build_dataset(test_df,terms_dict,protein_embeddings_all,protein_labels,fdl=fastdataloader)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BS, shuffle=False)
    valid_dataloader = DataLoader(val_dataset, batch_size=EVAL_BS, shuffle=False)
    train_labels = train_dataset.annotations
    valid_labels = val_dataset.annotations
    test_labels = test_dataset.annotations

    ## initialize model ##
    combine_model=Combine_Transformer(num_heads=num_heads,go_context=go_context_embeddings,embedding_vector=go_embeddings,device=device).to(device)  ## Cross attention model
    print(combine_model)
    train_labels = train_labels.detach().cpu().numpy()
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    optimizer = torch.optim.Adam(combine_model.parameters(), lr=5e-4)

    ### start training ###
    best_loss=100000.00
    for epoch in range(epoch_num):
        combine_model.train()
        train_loss = 0 ## DeepGO
        train_steps = int(math.ceil(len(train_labels) / TRAIN_BS))
        with ck.progressbar(length=train_steps, show_pos=True) as bar:
            for batch in train_dataloader:
                bar.update(1)
                batch_labels = batch['labels'].to(device)
                batch_features=batch['esm2_embeddings'].to(device)
                train_output=combine_model(batch_features).to(device)
                loss = F.binary_cross_entropy(train_output,batch_labels)  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()         
        train_loss /= train_steps

        print('validation')
        combine_model.eval()
        with torch.no_grad():
            valid_steps=int(math.ceil(len(valid_labels)/EVAL_BS))
            valid_loss = 0
            preds = []
            with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                for batch in valid_dataloader:
                    bar.update(1)
                    batch_labels = batch['labels'].to(device)
                    batch_features=batch['esm2_embeddings'].to(device)
                    valid_output=combine_model(batch_features)
                    batch_loss = F.binary_cross_entropy(valid_output,batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds = np.append(preds,valid_output.detach().cpu().numpy())
            valid_loss /= valid_steps
            roc_auc = compute_roc(valid_labels, preds)
            print(f'Epoch {epoch}: Loss - {train_loss} Valid loss - {valid_loss}, AUC - {roc_auc}')
        if valid_loss<best_loss:
            best_loss=valid_loss
            print('New record of loss on validation set: {:.4f}'.format(best_loss))
            torch.save(combine_model.state_dict(), model_save_file_path)

    print('Loading the best model')
    combine_model.load_state_dict(th.load(model_save_file_path))
    combine_model.eval()
    with torch.no_grad():
            test_loss=0
            test_steps=int(math.ceil(len(test_labels)/EVAL_BS))
            preds=[]
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch in test_dataloader:
                    bar.update(1)
                    batch_labels = batch['labels'].to(device)
                    batch_features=batch['esm2_embeddings'].to(device)
                    test_output=combine_model(batch_features).to(device)
                    batch_loss = F.binary_cross_entropy(test_output, batch_labels)
                    test_loss += batch_loss.detach().item()
                    preds.append(test_output.cpu().numpy())
                test_loss /= test_steps
            preds = np.concatenate(preds)
            roc_auc = compute_roc(test_labels, preds)
            print(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}')
    preds=list(preds)
    with Pool(32) as p:
        preds=p.map(partial(propagate_annots,go=go,terms_dict=terms_dict),preds)
    test_df['preds'] = preds
    test_df.to_pickle(f'{data_root}/{aspect}/predictions_esm2_context.pkl')

if __name__ == '__main__':
    main()