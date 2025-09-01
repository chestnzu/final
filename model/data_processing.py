import pandas as pd
from Bio import SeqIO
import torch
from defined_functions import EmbeddingTransform
import gensim
from owlready2 import *


goa_path="../data/goa_human.gaf"
sequence_path='../data/esm2650M_swissprot_human.pt'
### 数据预处理，找出所有包含有Annotation,且Annotation数量大于20的蛋白质
def load_filtered_protein_embeddings(
        goa_path:str,
        sequence_path:str,
        Annotation_threshold:int=5,
        GO_term_thresholad:int=20,
        IEA:bool=False):
    ## Read the GOA file
    goa=pd.read_csv(goa_path, sep="\t", comment='!', header=None) 
    ## deduplicate the GOA file
    goa_deduplicated=goa[[1, 4, 6]].drop_duplicates()
    ## Set column names
    goa_deduplicated.columns = ['DB_Object_ID', 'GO_ID', 'Evidence_Code']
    ## Filter out IEA evidence code if specified
    if not IEA:
        goa_deduplicated=goa_deduplicated[goa_deduplicated['Evidence_Code'] != 'IEA']
    ## filter proteins with more than threshold GO terms
    goa_deduplicated=goa_deduplicated[['DB_Object_ID', 'GO_ID']].drop_duplicates()
    go_count = goa_deduplicated.groupby('GO_ID')['DB_Object_ID'].count() 
    filtered_go_ids=go_count[go_count>=GO_term_thresholad].index.tolist() 
    goa_deduplicated =goa_deduplicated[goa_deduplicated['GO_ID'].isin(filtered_go_ids)]
    goa_deduplicated['GO_ID'] = goa_deduplicated['GO_ID'].str.replace('GO:', 'GO_')
    go_list = goa_deduplicated['GO_ID'].drop_duplicates().to_list() ## Get the list of GO terms

    protein_count = goa_deduplicated.groupby('DB_Object_ID')['GO_ID'].count() 
    filtered_protein_ids = protein_count[protein_count >= Annotation_threshold].index.tolist()
    goa_deduplicated = goa_deduplicated.loc[goa_deduplicated['DB_Object_ID'].isin(filtered_protein_ids)]

    filtered_protein_ids = goa_deduplicated['DB_Object_ID'].unique().tolist()

    file=torch.load(sequence_path)
    labels,sequences,_=file['labels'],file['sequences'],file['embeddings']
    sequence=pd.DataFrame({'ID':labels,'Sequence':sequences})
    sequence = sequence[sequence['ID'].isin(filtered_protein_ids)]

    GO_term_list=goa_deduplicated.loc[goa_deduplicated['DB_Object_ID'].isin(filtered_protein_ids)].\
        groupby('DB_Object_ID')["GO_ID"].apply(lambda x: ";".join(set(x))).reset_index().rename(columns={"GO_ID":"GO_Terms"})
    
    merged_df=sequence.merge(GO_term_list, left_on='ID', right_on='DB_Object_ID', how='inner') ##既有序列又有注释的蛋白质

    # 拆分为 ID 和 Tensor
    protein_id=merged_df['ID'].tolist()
    sequences=merged_df['Sequence'].tolist()
    annotated_go_terms=merged_df['GO_Terms'].tolist()
    return protein_id,sequences, annotated_go_terms, go_list


## 读取OWL2VEC生成的序列
embedding_path = '../data/model_vector/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'
def load_owl2vec_embeddings(embedding_path,onto_path):
    model=gensim.models.Word2Vec.load(embedding_path)
    onto=get_ontology(onto_path).load()
    classes = list(onto.classes())
    return model 




## ----- handle CAFA5 data -----
