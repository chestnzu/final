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
import random
import os


def set_seed(seed=42):
    """
    设置所有随机种子以确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 确保CUDA的确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


@ck.command()
@ck.option('--aspect','-asp',default='mf',type=ck.Choice(['mf','cc','bp']),
           help='GO aspect')
@ck.option('--deepgo2','-dp',is_flag=True,help='if deepgo2 model is used')
@ck.option('--fastdataloader','-fdl',is_flag=True,help='if use fast dataloader from deepgo2')
@ck.option('--protein_embedding_path','-ep',default='../data/esm_embeddings_3B_complete.pt')
@ck.option('--go_embedding_path','-ge',default='../data/go_all_embeddings.pt')
@ck.option('--seed','-s',default=42,type=int,help='Random seed for reproducibility')
@ck.option('--loss','-l',default='focal',type=ck.Choice(['bce','focal','asymmetric']),
           help='Loss function: bce, focal, or asymmetric (recommended for AUPR)')
@ck.option('--temperature','-t',is_flag=True,help='Use learnable temperature scaling for better calibration')


def main(aspect,deepgo2,fastdataloader,protein_embedding_path,go_embedding_path,seed,loss,temperature):
    # 设置随机种子以确保可复现性
    set_seed(seed)
    print(f'Random seed set to: {seed}')
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

    # 设置worker的随机种子以确保DataLoader的可复现性
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        _ = worker_id  # 使用worker_id防止警告

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True,
                                  worker_init_fn=seed_worker, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BS, shuffle=False,
                                 worker_init_fn=seed_worker, generator=g)
    valid_dataloader = DataLoader(val_dataset, batch_size=EVAL_BS, shuffle=False,
                                  worker_init_fn=seed_worker, generator=g)
    train_labels = train_dataset.annotations
    valid_labels = val_dataset.annotations
    test_labels = test_dataset.annotations

    ## initialize model ##
    combine_model=Combine_Transformer(num_heads=num_heads,go_context=go_context_embeddings,embedding_vector=go_embeddings,num_layers=num_layers,device=device,dropout=0.1,use_temperature=temperature).to(device)  ## Cross attention model
    if temperature:
        print(f"Using learnable temperature scaling (initial value: {combine_model.temperature.item():.2f})")

    ## initialize loss function ##
    if loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=1.6)
        print("Using Focal Loss (better for imbalanced data)")
    elif loss == 'asymmetric':
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8)
        print("Using Asymmetric Loss (optimized for AUPR) - Stable version")
    else:
        criterion = lambda pred, target: F.binary_cross_entropy(pred, target)
        print("Using Binary Cross Entropy Loss")

    ### early stop module ###
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    patience = 6
    min_delta = 1e-6
    #########################

    print(combine_model)
    train_labels = train_labels.detach().cpu().numpy()
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    optimizer = torch.optim.Adam(combine_model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  ### new
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
                loss = criterion(train_output, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()         
        train_loss /= train_steps
        scheduler.step()
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
                    batch_loss = criterion(valid_output, batch_labels)
                    valid_loss += batch_loss.detach().item()
                    preds = np.append(preds,valid_output.detach().cpu().numpy())
            valid_loss /= valid_steps
            roc_auc = compute_roc(valid_labels, preds)
            print(f'Epoch {epoch}: Loss - {train_loss} Valid loss - {valid_loss}, AUC - {roc_auc}')
        if valid_loss<best_loss-min_delta: ### -min_delta is for early stop
            best_loss=valid_loss
            print('New record of loss on validation set: {:.4f}'.format(best_loss))
            wait = 0 
            torch.save(combine_model.state_dict(), model_save_file_path)
        ### early stop ### 
        else:
            wait += 1     ### early stop
        
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    #########################
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
                    batch_loss = criterion(test_output, batch_labels)
                    test_loss += batch_loss.detach().item()
                    preds.append(test_output.cpu().numpy())
                test_loss /= test_steps
            preds = np.concatenate(preds)
            roc_auc = compute_roc(test_labels, preds)
            print(f'Valid Loss - {valid_loss}, Test Loss - {test_loss}, Test AUC - {roc_auc}')
            if temperature:
                print(f'Learned temperature: {combine_model.temperature.item():.4f}')
    preds=list(preds)
    with Pool(32) as p:
        preds=p.map(partial(propagate_annots,go=go,terms_dict=terms_dict),preds)
    test_df['preds'] = preds
    test_df.to_pickle(f'{data_root}/{aspect}/predictions_esm2_context_new.pkl')

if __name__ == '__main__':
    main()