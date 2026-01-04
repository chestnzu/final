from transformers import AutoTokenizer, AutoModel
import torch,obonet
from tqdm import tqdm
import click as ck
from Basic import *

@ck.command()
@ck.option('--go_file','-gf',default='../../data/go.obo',help='GO ontology file in OBO format')
@ck.option('--data_root','-dr',default='../../data',help='Data folder')

def main(go_file,data_root):
    go_embeddings = get_go_context(go_file, batch_size=16)
    torch.save(go_embeddings, "{}/go_context_embeddings.pt".format(data_root))
    print('GO context embeddings saved to {}/go_context_embeddings.pt'.format(data_root))

if __name__ == "__main__":
    main()