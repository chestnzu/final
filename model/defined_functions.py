import torch,obonet
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
from evaluation import *
from dataset_generating.basics import *
from sklearn.metrics import roc_curve, auc

class protein_loader(Dataset):
    def __init__(self,labels,esm2_embeddings):
#        self.protein_ids = protein_id
        self.esm2_embeddings = esm2_embeddings
        self.annotations = labels

    def __len__(self):
        return len(self.esm2_embeddings)
    
    def __getitem__(self, idx):
#        protein_id = self.protein_ids[idx]
        annotations = self.annotations[idx]
        esm2_embedding = self.esm2_embeddings[idx]
        return { 'labels': torch.as_tensor(annotations,dtype=torch.float32).clone().detach(),'esm2_embeddings': esm2_embedding}
    
def load_data(aspect,data_root,species=None):
    if species:
        train_data = pd.read_pickle(data_root + '/{}/train_data_{}.pkl'.format(aspect,species))
        valid_data = pd.read_pickle(data_root + '/{}/valid_data_{}.pkl'.format(aspect,species))
        test_data = pd.read_pickle(data_root + '/{}/test_data_{}.pkl'.format(aspect,species))
    else:
        train_data = pd.read_pickle(data_root + '/{}/train_data.pkl'.format(aspect))
        valid_data = pd.read_pickle(data_root + '/{}/valid_data.pkl'.format(aspect))
        test_data = pd.read_pickle(data_root + '/{}/test_data.pkl'.format(aspect))
    terms = pd.read_pickle(data_root + '/{}/terms.pkl'.format(aspect))['gos'].values.flatten() ## np.ndarray      
    terms_dict = {v: i for i, v in enumerate(terms)} ## (go_id:index)
    termidx={idx:term for term,idx in terms_dict.items()}  
    return train_data,valid_data,test_data,terms,terms_dict,termidx


def create_edge_index(onto_path,go_list,aspect):
    enc=LabelEncoder()
    onto = Ontology(onto_path, with_rels=True)
    aspect_map={'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    namespace_terms=onto.get_namespace_terms(aspect_map[aspect])
    go_list = list(set(go_list).intersection(namespace_terms))
    enc.fit(go_list)
    mapping={go_id: idx for idx, go_id in enumerate(enc.classes_)}
    nx_onto=obonet.read_obo(onto_path)
    onto_1=nx.relabel_nodes(nx_onto, mapping)
    go_list_digit=[idx for idx, _ in enumerate(enc.classes_)]
    edges=[(a,b) for a,b in onto_1.edges() if a in go_list_digit and b in go_list_digit]
    src,dst = zip(*edges)
    edge_index=torch.tensor([src,dst],dtype=torch.long)
    return edge_index, enc

def load_protein_embeddings(protein_ids,embedding,label):
    sequence_representations = []
    for pid in protein_ids:
        if pid in label:
            idx = label.index(pid)
            sequence_representations.append(embedding[idx].cpu())
        else:
            sequence_representations.append(torch.zeros(2560))
    embedding_batch = torch.stack(sequence_representations)
    return embedding_batch


def build_dataset(dataset, term_dict,protein_embeddings, protein_labels,exp_only=False,fdl=False):
    protein_id = list(dataset['proteins'].values)
    esm2_embeddings=load_protein_embeddings(protein_id, protein_embeddings, protein_labels)
    labels = torch.zeros((len(dataset), len(term_dict)), dtype=torch.float32)
    # ----------- 处理 annotations ----------
    for i,row in enumerate(dataset.itertuples()):
        if exp_only:
            for go in row.exp_annotations:
                if go in term_dict:
                    gid=term_dict[go]
                    labels[i,gid]=1
        else:
            for go in row.prop_annotations:
            #for go in row.propagate_annotation:
                if go in term_dict:
                    gid=term_dict[go]
                    labels[i,gid]=1
    if fdl == True:
        return esm2_embeddings,labels
    else:
        dataset = protein_loader(labels,esm2_embeddings.cpu())
        return dataset


def protein_go_contrastive_loss(protein_vecs, go_vecs, temperature=0.1):
    """
    蛋白质→GO单向对比损失
    - 每个蛋白质与一个GO术语正样本配对
    - 其他GO术语作为负样本
    """
    protein_vecs = F.normalize(protein_vecs, dim=1)
    go_vecs = F.normalize(go_vecs, dim=1)
    
    # 蛋白质查询GO空间
    logits = protein_vecs @ go_vecs.T  # [batch_size, batch_size]
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    loss = F.cross_entropy(logits / temperature, labels)
    return loss


def load_normal_forms(go_file, terms_dict):
    """
    Parses and loads normalized (using Normalize.groovy script)
    ontology axioms file
    Args:
        go_file (string): Path to a file with normal forms
        terms_dict (dict): Dictionary with GO classes that are predicted
    Returns:
        
    """
    nf1 = []
    nf2 = []
    nf3 = []
    nf4 = []
    relations = {}
    zclasses = {}
    
    def get_index(go_id):
        if go_id in terms_dict:
            index = terms_dict[go_id]
        elif go_id in zclasses:
            index = zclasses[go_id]
        else:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
            index = zclasses[go_id]
        return index

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]
                
    with open(go_file) as f:
        for line in f:
            line = line.strip().replace('_', ':')
            if line.find('SubClassOf') == -1:
                continue
            left, right = line.split(' SubClassOf ')
            # C SubClassOf D
            if len(left) == 10 and len(right) == 10:
                go1, go2 = left, right
                nf1.append((get_index(go1), get_index(go2)))
            elif left.find('and') != -1: # C and D SubClassOf E
                go1, go2 = left.split(' and ')
                go3 = right
                nf2.append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find('some') != -1:  # R some C SubClassOf D
                rel, go1 = left.split(' some ')
                go2 = right
                nf3.append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find('some') != -1: # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(' some ')
                nf4.append((get_index(go1), get_rel_index(rel), get_index(go2)))
    return nf1, nf2, nf3, nf4, relations, zclasses

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_metrics(test_df, go, terms_dict, terms, ont, eval_preds):
    ## test_df: dataframe with columns 'proteins', 'accessions', 'sequences', 'annotations', 'species','exp_annotations', 'propagate_annotation'
    ## eval_preds: numpy array of shape (number of proteins, number of GO terms),
    ## go: Ontology object
    ## term_dict: {go_id: column_index}
    ## terms: list of go_ids corresponding to columns in eval_preds
    ## ont: ontology aspect, one of 'mf','bp','cc'

    labels = np.zeros((len(test_df), len(terms_dict)), dtype=np.float32) ## len(test_df) == number of proteins
                                                                         ## len(terms_dict) == number of GO terms
                                                                         
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1   ## for each protein, mark the presence of propagated GO terms
    
    total_n = 0
    total_sum = 0
    for go_id, i in terms_dict.items(): ## iterate over all GO columns
        pos_n = np.sum(labels[:, i]) ## number of positive samples for this GO term
        if pos_n > 0 and pos_n < len(test_df):
            total_n += 1
            roc_auc  = compute_roc(labels[:, i], eval_preds[:, i])
            total_sum += roc_auc

    avg_auc = total_sum / total_n
    
    print('Computing Fmax')
    fmax = 0.0
    tmax = 0.0
    wtmax = 0.0
    wfmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    go_set = go.get_namespace_terms(NAMESPACES[ont]) ## get all GO terms in the specific ontology
    go_set.remove(FUNC_DICT[ont]) ## remove the root term
    labels = test_df['prop_annotations'].values   ## get propagated annotations for each protein
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels)) ## filter out annotations not in the go set
    for t in range(0, 101): ## from 0 to 1 with step size 0.01
        threshold = t / 100.0 ## threshold for deciding whether a GO term is predicted to be annotated to a protein
        preds = [set() for _ in range(len(test_df))] ## initialize empty set for each protein
        for i in range(len(test_df)):
            annots = set()
            above_threshold = np.argwhere(eval_preds[i] >= threshold).flatten()
            for j in above_threshold:
                annots.add(terms[j])             
            if t==0:
                preds[i] = annots
                continue
            preds[i] = annots
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))   

 ## filter out predictions not in the go set
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf  = evaluate_annotations(go, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            avgic = avg_ic
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    

    return fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic    # fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic



def evaluate_annotations(go, real_annots, pred_annots):
    """
    Computes Fmax, Smin, WFmax and Average IC
    Args:
       go (utils.Ontology): Ontology class instance with go.obo
       real_annots (set): Set of real GO classes ## list of lists, each set for a protein, len(list) = number of GO terrms
       pred_annots (set): Set of predicted GO classes
    """
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = set(pred_annots[i]) - tp
        fn = set(real_annots[i]) - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            if tpic + fpic > 0:
                wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf



def load_deepgo2_data(data_root,aspect): ## only works for deepgo2 original data, because their df contains esm2 embeddings
    train_data = pd.read_pickle(data_root + '/{}/train_data.pkl'.format(aspect))
    valid_data = pd.read_pickle(data_root + '/{}/valid_data.pkl'.format(aspect))
    test_data = pd.read_pickle(data_root + '/{}/test_data.pkl'.format(aspect))

    all_data=pd.concat([train_data,valid_data,test_data])
    protein_labels=all_data['proteins'].tolist()
    protein_embeddings=torch.from_numpy(np.stack(all_data['esm2'].values))
    return protein_labels, protein_embeddings


import torch

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    