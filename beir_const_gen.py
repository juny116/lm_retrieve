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
    device = torch.device(config['device'])

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(len(corpus), len(queries), len(qrels))
    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    model = model.to(device)

    sents = []
    for k, v in tqdm(corpus.items()):
        sents.append(tokenizer.encode(v['text'], truncation=True, max_length=384) + [-1, k])
    trie = Trie(sents)
    with open(f'results/{dataset}_trie.pkl', 'wb') as f:
        pickle.dump(trie.trie_dict, f)

    # with open(f'results/{dataset}_trie.pkl', 'rb') as f:
    #     trie_dict = pickle.load(f)
    # trie = Trie.load_from_dict(trie_dict)
    # print(trie.get([]))
    template = config['templates']['template']
    print(template)
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        start = time()
        input_str = template.replace('[QUERY]', queries[q_id])
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.size(1)
        # print(input_str)
        # print(input_len)

        force_words_ids = [trie.get([])]
        if len(force_words_ids[0]) > 1:
            force_words_ids = [[[ids] for ids in force_words_ids[0]]]
        for j in range(384):
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                force_words_ids=force_words_ids,
                num_beams=6,
                num_return_sequences=1,
                remove_invalid_values=True,
                use_cache=True,
            )
            # print(outputs[0][1:].tolist())
            # print(tokenizer.decode(outputs[0][1:], skip_special_tokens=True))
            input_ids = torch.cat((input_ids[0], outputs[0][1:]), dim=0).expand(1, -1)
            # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
            # print(input_ids[0][input_len:].tolist())
            force_words_ids = [trie.get(input_ids[0][input_len:].tolist())]
            if len(force_words_ids[0]) > 1:
                force_words_ids = [[[ids] for ids in force_words_ids[0]]]
            # print(force_words_ids)
            if force_words_ids[0] == [-1]:
                break
        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        force_words_ids = [trie.get(input_ids[0][input_len:].tolist() + [-1])]
        print(force_words_ids)
        print(c)
        print(time() - start)
        if i > 30:
            break

if __name__ == "__main__":
    main()