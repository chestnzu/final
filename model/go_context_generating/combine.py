import pandas as pd
import click as ck
from Basic import *

@ck.command()
@ck.option('--data-root', '-dr', default='data',
           help='Data folder')
@ck.option('--embedding_path', '-ep', default='../../data/pre_trained_model/ontology.embeddings')


def main(data_root,embedding_path):
    go_context=torch.load('{}/go_context_embeddings.pt'.format(data_root))
    go_embeddings=load_go_embeddings(embedding_path,go_context['terms'])
    go_all_embeddings={'terms':go_context['terms'],'context_embeddings':go_context['embeddings'].detach().cpu().numpy(),
                       'go_embeddings':go_embeddings}
    torch.save(go_all_embeddings,'{}/go_all_embeddings.pt'.format(data_root))
    print('GO all embeddings saved to {}/go_all_embeddings.pt'.format(data_root))


if __name__ == "__main__":
    main()