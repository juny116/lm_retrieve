import os, random
import logging
import json, pickle
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
from beir.retrieval import models
from trie import Trie
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from time import time


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset

    dataset = config['task']
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = '/datasets/datasets/beir'
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(len(corpus), len(queries), len(qrels))
    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    # for k, v in queries.items():
    #     print(v)
    #     break
    # model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    sents = []
    for k, v in tqdm(corpus.items()):
        sents.append(tokenizer.encode(v['text'], truncation=True, max_length=384))
    # sents = [tokenizer.encode(doc['text'], truncation=True, max_length=384) for doc in corpus.values()]
    start = time()
    trie = Trie(sents)
    print(time() - start)
    with open(f'{dataset}_trie.pkl', 'wb') as f:
        pickle.dump(trie.trie_dict, f)
    
if __name__ == "__main__":
    main()