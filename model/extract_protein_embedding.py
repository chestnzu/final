import esm
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
## 提取蛋白质的ESM-2嵌入
import click


@click.command()
@click.option( '--input-path','-in',type=click.Path(exists=True, dir_okay=False), required=True,
    help='Path to the input sequence file')
@click.option('--output-path','-out',type=click.Path(dir_okay=False),required=True,
    help='Path to save the output embeddings')
@click.option('--maxlen','-mlen',type=int,default=512,show_default=True,
    help='Maximum sequence length')
@click.option('--model-name','-m',
    type=click.Choice([
        'esm2_t6_8M_UR50D',
        'esm2_t12_35M_UR50D',
        'esm2_t30_150M_UR50D',
        'esm2_t33_650M_UR50D',
        'esm2_t36_3B_UR50D'
    ]),
    default='esm2_t33_650M_UR50D',
    show_default=True,
    help='ESM-2 model name'
)



def main(input_path,output_path,model_name,maxlen):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the ESM-2 model and alphabet
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print('load model successfully')
    batch_converter = alphabet.get_batch_converter()
    seqs_info = {record.id: str(record.seq)[:maxlen] for record in SeqIO.parse(input_path, "fasta")}
    items = list(seqs_info.items())

    batch_size = 2
    save_every = 500  # 每 500 个 batch 保存一次
    part = 1            # 保存文件编号

    buffer_labels = []
    buffer_seqs = []
    sequence_representations = []

    for batch in tqdm(range(0, len(seqs_info),batch_size),desc="Extracting embeddings"):
        batch_seqs = items[batch:batch+batch_size]
        labels, strs, tokens = batch_converter(batch_seqs)
        tokens=tokens.to(device)
        batch_lens = (tokens != alphabet.padding_idx).sum(1)
        with torch.inference_mode():
            results = model(tokens, repr_layers=[36],return_contacts=False)
        token_representations = results["representations"][36]
        for i, tokens_len in enumerate(batch_lens):
            buffer_labels.append(batch_seqs[i][0])
            buffer_seqs.append(batch_seqs[i][1])
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        if (batch // batch_size + 1) % save_every == 0:
            save_file = f"{output_path}part{part}.pt"
            torch.save({
                "labels": buffer_labels,
                "sequences": buffer_seqs,
                "embeddings": sequence_representations
            }, save_file)

            print(f"Saved: {save_file}")
            part += 1

            # 清空 buffer
            buffer_labels = []
            buffer_seqs = []
            sequence_representations = []

    # ✔ 保存最后不足 1000 batch 的部分
    if buffer_labels:
        save_file = f"{output_path}part{part}.pt"
        torch.save({
            "labels": buffer_labels,
            "sequences": buffer_seqs,
            "embeddings": sequence_representations
        }, save_file)

        print(f"Saved: {save_file}")

    print("All embeddings saved successfully.")

    # results = {
    #     "labels": list(seqs_info.keys()),
    #     "sequences": list(seqs_info.values()),
    #     "embeddings": sequence_representations
    # }
    # torch.save(results, output_path)
    # print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    main()