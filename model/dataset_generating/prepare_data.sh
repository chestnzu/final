#!/bin/bash
data_root=../../data

echo "preparing dataset"
python uniprot2df.py -sf $data_root/uniprot_sprot.dat.gz -o $data_root/swissprot.pkl

echo "creating FASTA file as background"
python gen_fasta.py -df $data_root/swissprot.pkl -o $data_root/swissprot.fa -n all

echo "Creating Diamond database and compute similarities"
diamond makedb --in $data_root/swissprot.fa --db $data_root/swissprot.dmnd
diamond blastp --very-sensitive -d $data_root/swissprot.dmnd -q $data_root/swissprot.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/swissprot.sim

echo "splitting dataset"
python train_dataset_split.py -s 9606

echo 'generating fasta file for esm2'
python gen_fasta.py -df $data_root/swissprot.pkl -o $data_root/swissprot_9606.fa -n 9606


### if you want to train your own esm2 model, uncomment the following lines
# echo 'training esm2 model'
# python ../extract_protein_embedding.py --input_path=$data_root/swissprot_9606.fa --output_path=$data_root/9606vector.pt --max_len=1024 --model_name=esm2_t33_650M_UR50D