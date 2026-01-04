import numpy as np
import pandas as pd
from dataset_generating.basics import *
import click as ck
from dataset_generating.basics import Ontology
from defined_functions import *

@ck.command()
@ck.option(
    '--data-root', '-dr', default='../data/dataset/',
    help='Data folder')
@ck.option(
    '--onto', '-onto', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--model-name', '-m', help='Prediction model name')
# @ck.option(
#     '--test-data-name', '-td', default='test', type=ck.Choice(['test']),
    # help='Test data set name')


def main(data_root, onto, model_name):
    # train_data_file = f'{data_root}/{onto}/train_data_yi.pkl'
    # valid_data_file = f'{data_root}/{onto}/valid_data_yi.pkl'    
    test_data_file = f'{data_root}/{onto}/predictions_esm2_context.pkl'
    train_data_file = f'{data_root}/{onto}/train_data.pkl'
    valid_data_file = f'{data_root}/{onto}/valid_data.pkl'    
#    test_data_file = f'{data_root}/{onto}/test_data.pkl'



    terms_file = f'{data_root}/{onto}/terms.pkl'
    if 'dataset' not in data_root:
        go = Ontology(f'{data_root}/go.obo', with_rels=True)
    else:
        go = Ontology(f'{data_root}/../go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df= pd.read_pickle(test_data_file)

    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values

    test_annotations = list(map(lambda x: set(x), test_annotations))
    go.calculate_ic(annotations + test_annotations)

    ics = {}
    for _, term in enumerate(terms):
        ics[term] = go.get_ic(term)

    eval_preds = []

    for _,row in enumerate(test_df.itertuples()):
        preds = row.preds
        eval_preds.append(preds)       

    eval_preds = np.concatenate(eval_preds).reshape(-1, len(terms))
    fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic = compute_metrics(test_df,go,terms_dict,terms,onto,eval_preds)
    print(model_name, onto)
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    print(f'WFmax: {wfmax:0.3f}, threshold: {wtmax}')
    print(f'AUC: {avg_auc:0.3f}')
    print(f'AUPR: {aupr:0.3f}')
    print(f'AVGIC: {avgic:0.3f}')



if __name__ == '__main__':
    main()
