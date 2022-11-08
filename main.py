from datasets import load_dataset
from transformers import AutoModel
from elasticsearch import Elasticsearch
from rank_bm25 import BM25Okapi


d = load_dataset('BeIR/msmarco', 'corpus', cache_dir='/datasets/datasets/huggingface/beir')

print(d[0])
# from rank_bm25 import BM25Okapi

# def tokenizer(sent):
#   return sent.split(" ")

# tokenized_corpus = [tokenizer(doc) for doc in corpus]

# bm25 = BM25Okapi(tokenized_corpus)
# m = AutoModel.from_pretrained('bert-base-uncased', cache_dir='/datasets/datasets/huggingface/models')
