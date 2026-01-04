from transformers import AutoTokenizer, AutoModel
import torch,obonet
from tqdm import tqdm
import gensim
import numpy as np

def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def load_go_embeddings(embedding_path,go_terms):
    model=gensim.models.Word2Vec.load(embedding_path)
    full_vectors=[]
    for term in go_terms:
        vector=model.wv.get_vector("http://purl.obolibrary.org/obo/"+term.replace(':','_'))
        full_vectors.append(torch.tensor(vector))
    matrix = np.array(full_vectors)
    return matrix



def get_go_context(go_file,batch_size,model_name="neuml/pubmedbert-base-embeddings"):
    go_file=obonet.read_obo(go_file)
    terms=list(go_file.nodes)
    texts = []
    for go_term in terms:
        obj=go_file.nodes[go_term]
        name = obj.get("name", "")
        definition = obj.get("def", "")
        namespace = obj.get("namespace", "")
        go_context = f"GOId: {go_term}; Name: {name}; Namespace: {namespace}; Definition: {definition}"
        texts.append(go_context)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt",max_length=256).to(device)
            outputs = model(**inputs)
            batch_emb = meanpooling(outputs, inputs['attention_mask'])  # [batch, dim]
            all_embeddings.append(batch_emb.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [num_terms, dim]
    assert len(terms) == len(all_embeddings)
    return {"terms": terms,"embeddings": all_embeddings}
