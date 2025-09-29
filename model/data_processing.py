import pandas as pd
from Bio import SeqIO
import torch
from defined_functions import EmbeddingTransform
import gensim
from owlready2 import *

onto_path='../data/go-basic.owl'
goa_path="../data/goa_human.gaf"
sequence_path='../data/esm2650M_swissprot_human.pt'


## 读取OWL2VEC生成的序列
embedding_path = '../data/model_vector/owl2vec_go_basic.embeddings'

def load_owl2vec_embeddings(embedding_path):
    model=gensim.models.Word2Vec.load(embedding_path)
    return model 





