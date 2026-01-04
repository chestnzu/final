#!/bin/bash
data_root=../../data


echo 'generating GO embedding using OWL2VEC*'
(
  cd ../../../OWL2Vec-Star/ || exit 1
  owl2vec_star standalone --config_file default.cfg
)


echo "generating GO context embeddings"
python go_context_embedding.py -gf $data_root/go.obo -dr $data_root

echo "combining GO context embeddings with existing go embedding from owl2vec star"
python combine.py -dr $data_root 

echo "complete"