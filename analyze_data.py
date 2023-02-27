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
import numpy as np


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

    template = config['templates']['template']

    for q in queries:
        print(q, queries[q])
        break
    for c in corpus:
        print(c, corpus[c]['text'])
        break

    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'], max_length=2048)

    q_token_lenghts = []
    qt_token_lenghts = []
    for q in tqdm(queries):
        q_token_lenghts.append(len(tokenizer.encode(queries[q], max_length=2048, truncation=True)))
        input_str = template.replace('[QUERY]', queries[q])
        qt_token_lenghts.append(len(tokenizer.encode(input_str, max_length=2048, truncation=True)))

    c_token_lenghts = []
    for c in tqdm(corpus):
        c_token_lenghts.append(len(tokenizer.encode(corpus[c]['text'], max_length=2048, truncation=True)))

    hist, bins = np.histogram(c_token_lenghts, bins=20)
    print(hist)
    print(bins)
    print(f"{np.mean(c_token_lenghts)} {np.median(c_token_lenghts)} {np.std(c_token_lenghts)}")
    print(f"{np.mean(q_token_lenghts)} {np.median(q_token_lenghts)} {np.std(q_token_lenghts)}")
    print(f"{np.mean(qt_token_lenghts)} {np.median(qt_token_lenghts)} {np.std(qt_token_lenghts)}")
if __name__ == "__main__":
    main()