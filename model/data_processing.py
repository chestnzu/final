import pandas as pd
from Bio import SeqIO
import torch
from defined_functions import EmbeddingTransform
import gensim
from owlready2 import *


def preprocessing_goa(data_path,evidence_code_type='all'):
    """
    Preprocess the GOA file to create a mapping of Protein IDs to GO Terms.
    """
    # Read the GOA file
    df = pd.read_csv(data_path, sep='\t', comment='!', header=None)
    
    # Set column names
    # df.columns = ['DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID', 
    #               'DB_Reference', 'Evidence_Code', 'With_or_From', 'Aspect', 
    #               'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 
    #               'Taxon_ID', 'Date', 'Assigned_By']
    
    # Select relevant columns
    df = df[[1, 4, 6]]
    df.drop_duplicates(inplace=True)  # Remove duplicates
    df.columns = ['DB_Object_ID', 'GO_ID', 'Evidence_Code']
    if evidence_code_type == 'all':
        pass
    elif evidence_code_type == 'non-electronic':
        df=df[df['Evidence_Code'] != 'IEA']     
    elif evidence_code_type == 'experimental':
        df = df[df['Evidence_Code'].isin(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP'])]
    elif evidence_code_type == 'experimental and phylogenetic':
        df = df[df['Evidence_Code'].isin(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'IBA','IBD', 'IKR', 'IRD'])]

    # Group by Protein ID and join GO Terms
    go_map = df.groupby('DB_Object_ID')["GO_ID"].apply(lambda x: ";".join(set(x))).reset_index()
    go_map.columns = ["ProteinID", "GO_Terms"]
    return go_map
    # Save to TSV file
    # go_map.to_csv("../data/train_labels.tsv", sep="\t", index=False)

MAXLEN=500
def preprocessing_sequence(data_path,MAXLEN=MAXLEN):
    """ truncate sequence to MAXLEN and extract UniProt ID """
    data=[]
    for record in SeqIO.parse(data_path, "fasta"):
        header = record.description  # 全部描述行
        seq = str(record.seq)
        if len(seq) > MAXLEN:
            seq = seq[:MAXLEN]
        try:
            uniprot_id = header.split("|")[1]
        except:
            uniprot_id = record.id
        data.append((uniprot_id,seq))
    return data


goa_path="../data/goa_human.gaf"
sequence_path='../data/train_sequences.tsv'
embedding_path='../data/model_vector/esm_swissprot_650U_500.pt'
### 数据预处理，找出所有包含有Annotation,且Annotation数量大于20的蛋白质
def load_filtered_protein_embeddings(
        goa_path:str,
        sequence_path:str,
        threshold:int=20,
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
    goa_term=goa_deduplicated.groupby('DB_Object_ID')['GO_ID'].count()>threshold
    filtered_protein_ids=goa_term[goa_term].index.tolist()  ## 输出满足条件的蛋白质ID
    sequence = pd.read_csv(sequence_path, sep='\t')    
    sequence = sequence[sequence['ID'].isin(filtered_protein_ids)]

    GO_term_list=goa_deduplicated.loc[goa_deduplicated['DB_Object_ID'].isin(filtered_protein_ids)].\
        groupby('DB_Object_ID')["GO_ID"].apply(lambda x: ";".join(set(x))).reset_index().rename(columns={"GO_ID":"GO_Terms"})
    
    merged_df=sequence.merge(GO_term_list, left_on='ID', right_on='DB_Object_ID', how='inner')
    # 拆分为 ID 和 Tensor
    protein_id=merged_df['ID'].tolist()
    sequences=merged_df['Sequence'].tolist()
    annotated_go_terms=merged_df['GO_Terms'].tolist()
    return protein_id,sequences, annotated_go_terms

## 读取OWL2VEC生成的序列
embedding_path = '../data/model_vector/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'
def load_owl2vec_embeddings(embedding_path,onto_path):
    model=gensim.models.Word2Vec.load(embedding_path)
    onto=get_ontology(onto_path).load()
    classes = list(onto.classes())
    return classes, model 