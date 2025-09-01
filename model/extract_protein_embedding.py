import esm
import torch
import numpy as np
import argparse
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

## 提取蛋白质的ESM-2嵌入

parser = argparse.ArgumentParser(description='Extract ESM-2 embeddings for protein sequences')
parser.add_argument('--input_path', type=str, required=True, help='Path to the sequence file')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings')
parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D', help='ESM-2 model name')

args = parser.parse_args()

input_path=args.input_path
output_path=args.output_path
model_name=args.model_name
maxlen=args.max_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    # Load the ESM-2 model and alphabet
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print('load model successfully')
    batch_converter = alphabet.get_batch_converter()
    seqs_info = {record.id.split('|')[1]: str(record.seq)[:maxlen] for record in SeqIO.parse(input_path, "fasta")}
    sequence_representations = []
    items = list(seqs_info.items())
    for batch in tqdm(range(0, len(seqs_info), 100),desc="Extracting embeddings"):
        batch_seqs = items[batch:batch+10]
        labels, strs, tokens = batch_converter(batch_seqs)
        tokens=tokens.to(device)
        batch_lens = (tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33],return_contacts=False)
        token_representations = results["representations"][33]
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    results = {
        "labels": list(seqs_info.keys()),
        "sequences": list(seqs_info.values()),
        "embeddings": sequence_representations
    }
    torch.save(results, output_path)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    run()