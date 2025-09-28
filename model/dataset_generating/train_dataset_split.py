import os
import sys
sys.path.append('.')

import click as ck
import numpy as np
import pandas as pd
from collections import Counter, deque
from basics import Ontology, FUNC_DICT, NAMESPACES, MOLECULAR_FUNCTION, BIOLOGICAL_PROCESS, CELLULAR_COMPONENT, HAS_FUNCTION,generate_go_term_list

@ck.command()
@ck.option(
    '--go-file', '-gf', default='../data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='../data/swissprot.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--sim-file', '-sf', default='../data/swissprot_9606.sim',
    help='Sequence similarity generated with Diamond')
@ck.option(
    '--species','-s', default='all',
    help='species to include,e.g., 9606')

def main(go_file, data_file, sim_file, species):
    go = Ontology(go_file, with_rels=True)
    df = pd.read_pickle(data_file)
    proteins = set(df['proteins'].values)
    if species != 'all':
        df = df[df['species'] == species]
        df = df.reset_index(drop=True)
    
    print("DATA FILE" ,len(df))
        

    for ont in ['cc', 'bp', 'mf']:
        cnt = Counter()
        f1=generate_go_term_list(go_file,ont)
        index = []
        for i, row in enumerate(df.itertuples()):
            ok = False
            for term in row.propagate_annotation:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            if ok:
                index.append(i)

        del cnt[FUNC_DICT[ont]] # Remove top term
        tdf = df.iloc[index]
        terms = list(cnt.keys())


        print(f'Number of {ont} terms {len(terms)}')
        print(f'Number of {ont} proteins {len(tdf)}')
    
        # terms_df = pd.DataFrame({'gos': terms})
        # terms_df.to_pickle(f'data/{ont}/terms.pkl')

        # Split train/valid/test
        proteins = tdf['proteins']
        prot_set = set(proteins)
        prot_idx = {v:k for k, v in enumerate(proteins)}
        sim = {}
        train_prots = set()
        with open(sim_file) as f:
            for line in f:
                it = line.strip().split('\t')
                p1, p2, _ = it[0], it[1], float(it[2]) / 100.0
                if p1 == p2:
                    continue
                if p1 not in prot_set or p2 not in prot_set:
                    continue
                if p1 not in sim:
                    sim[p1] = []
                if p2 not in sim:
                    sim[p2] = []
                sim[p1].append(p2)
                sim[p2].append(p1)

        used = set()
        def dfs(prot):
            stack = deque()
            stack.append(prot)
            used.add(prot)
            prots = []
            while len(stack) > 0:
                prot = stack.pop()
                prots.append(prot)
                used.add(prot)
                if prot in sim:
                    for p in sim[prot]:
                        if p not in used:
                            used.add(p)
                            stack.append(p)
            return prots

        groups = []
        for p in proteins:
            if p not in used:
                group = dfs(p)
                groups.append(group)
        print(len(proteins), len(groups))
        index = np.arange(len(groups))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_n = int(len(groups) * 0.9)
        valid_n = int(train_n * 0.9)
    
        train_index = []
        valid_index = []
        test_index = []
        for idx in index[:valid_n]:
            for prot in groups[idx]:
                train_index.append(prot_idx[prot])
        for idx in index[valid_n:train_n]:
            for prot in groups[idx]:
                valid_index.append(prot_idx[prot])
        for idx in index[train_n:]:
            for prot in groups[idx]:
                test_index.append(prot_idx[prot])
                
        train_index = np.array(train_index)
        valid_index = np.array(valid_index)
        test_index = np.array(test_index)

        train_df = tdf.iloc[train_index]
        train_df.to_pickle(f'../data/dataset/{ont}/train_data.pkl')
        
        valid_df = tdf.iloc[valid_index]
        valid_df.to_pickle(f'../data/dataset/{ont}/valid_data.pkl')
        test_df = tdf.iloc[test_index]
        test_df.to_pickle(f'../data/dataset/{ont}/test_data.pkl')
        f1.to_pickle(f'../data/dataset/{ont}/terms.pkl')
        print(f'Train/Valid/Test proteins for {ont} {len(train_df)}/{len(valid_df)}/{len(test_df)}')
        print(f'terms {len(terms)}')

                
if __name__ == '__main__':
    main()