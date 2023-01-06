from mteb.tasks import SciFact, NQ, MSMARCOv2, MSMARCO, DBPedia, FEVER, QuoraRetrieval
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import random
from tqdm import tqdm

random.seed(42)
max_cnt = 3

cuda = torch.device('cuda:1')
model_name = "google/flan-t5-xl"
# template = "Generate a document that answers the following question. "
# template = "Generate a wikipedia document that answers the following question. "
# template = "Generate a document that answers the following Question. Question: [QUERY]. Output: "
template = f"Generate a wikipedia document that answers the following Question. Question: [QUERY]. Output: "
# template = f"Generate a synthetic wikipedia document that answers the following Question. Question: [QUERY]. Output:"
# template = f"Generate a synthetic document that offers evidence for the following Claim. Claim: [QUERY]. Output: "
# "allenai/tk-instruct-11b-def-pos"
# "allenai/tk-instruct-3b-def"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(cuda)

dataset = NQ()
# dataset = SciFact()
dataset.load_data(eval_splits=['test'])
rel = dataset.relevant_docs['test']
queries = dataset.queries['test']
corpus = dataset.corpus['test']

results = []
cnt = 0
for q_id, c in tqdm(rel.items()):
    c_id = next(iter(c))
    input_str = template.replace('[QUERY]', queries[q_id])

    input_ids = tokenizer.encode(input_str, return_tensors="pt")
    input_ids = input_ids.to(cuda)

    output = model.generate(input_ids,
        do_sample=True,                             
        max_length=500, 
        top_k=50, 
        top_p=0.95, 
        min_length=150,
        num_return_sequences=3
    )

    result = {'q_id': q_id, 'c_id': c_id, 'query': queries[q_id], 'gt': corpus[c_id]['text'], 'output': [tokenizer.decode(output[j], skip_special_tokens=True) for j in range(3)]}
    results.append(result)

    cnt += 1
    if cnt >= max_cnt:
        break

with open('nq_4.pkl', 'wb') as f:
    pickle.dump(results, f)