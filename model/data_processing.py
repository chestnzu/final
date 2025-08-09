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
def preprocess_goa_data(goa_path,threshold=20,IEA=False):
    goa=pd.read_csv(goa_path, sep="\t", comment='!', header=None)
    goa_deduplicated=goa[[1, 4, 6]].drop_duplicates()
    goa_deduplicated.columns = ['DB_Object_ID', 'GO_ID', 'Evidence_Code']
    if not IEA:
        goa_deduplicated=goa_deduplicated[goa_deduplicated['Evidence_Code'] != 'IEA']
    goa_term=goa_deduplicated.groupby('DB_Object_ID')['GO_ID'].count()>threshold
    goa_term=goa_term[goa_term].index.tolist()
    annotated_go_terms=goa_deduplicated.loc[goa_deduplicated['DB_Object_ID'].isin(goa_term), 'GO_ID'].unique().tolist()
    return goa_term,annotated_go_terms



goa_term_ids=preprocess_goa_data(goa_path)
def load_filtered_protein_embeddings(sequence_path, embedding_path, goa_term_ids):
    """
    根据 GOA term 过滤蛋白质 embedding，返回 ID 和 embedding Tensor。
    """
    # 读取蛋白质 ID 列表（顺序必须和 embedding 对应）
    sequence = pd.read_csv(sequence_path, sep='\t', usecols=[0])
    protein_name = sequence['ID'].tolist()
    # 加载对应的 ESM embedding
    protein_embedding = torch.load(embedding_path)  # List[Tensor]
    # Zip 成 (id, embedding)
    embedding_with_id = list(zip(protein_name, protein_embedding))
    # 过滤：只保留在 GOA term 中的蛋白
    filtered_embeddings = [(pid, vec) for pid, vec in embedding_with_id if pid in goa_term_ids]
    # 拆分为 ID 和 Tensor
    protein_ids, protein_embeddings = zip(*filtered_embeddings)
    protein_embeddings = torch.stack(protein_embeddings)
    return protein_ids, protein_embeddings


## 读取OWL2VEC生成的序列
embedding_path = '../data/model_vector/owl2vec_go_basic.embeddings'
onto_path='../data/go-basic.owl'
def load_owl2vec_embeddings(embedding_path,onto_path):
    model=gensim.models.Word2Vec.load(embedding_path)
    onto=get_ontology(onto_path).load()
    classes = list(onto.classes())
    return classes, model 