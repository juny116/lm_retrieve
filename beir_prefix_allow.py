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
from trie import Trie
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from time import time
import evaluate


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
    max_gen = config['max_gen']

    metric = evaluate.load('ndcg.py')

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(len(corpus), len(queries), len(qrels))
    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    model = model.to(device)

    sents = []
    for k, v in tqdm(corpus.items()):
        sents.append([0] + tokenizer.encode(v['text'], truncation=True, max_length=128) + [-1, k])
    trie = Trie(sents)

    with open(f'results/{dataset}_128_trie.pkl', 'wb') as f:
        pickle.dump(trie.trie_dict, f)

    # with open(f'results/{dataset}_trie.pkl', 'rb') as f:
    #     trie_dict = pickle.load(f)
    # trie = Trie.load_from_dict(trie_dict)

    def prefix_allowed_fn(batch_id, sent):
        # print(batch_id, sent)
        sent = sent.tolist()
        trie_out = trie.get(sent)
        if trie_out == [-1]:
            trie_out = []
        # print(trie_out)
        return trie_out

    # print(trie.get([]))
    template = config['templates']['template']
    print(template)
    gen_results = set()
    total_num = 0
    correct_num = 0
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        if i > max_gen:
            break
        input_str = template.replace('[QUERY]', queries[q_id])
        input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.size(1)
        # print(input_str)
        # print(input_len)

        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            prefix_allowed_tokens_fn=prefix_allowed_fn,
            num_beams=10,
            num_return_sequences=10,
            remove_invalid_values=True,
            use_cache=True,
        )

        # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        cid_list = []
        for output in outputs:
            out_list = output.tolist()
            temp = [out_list[0]]
            for out in out_list[1:]:
                if out == 0:
                    break
                temp.append(out)
            cid_list.append(trie.get(temp + [-1])[0])
        # c_id = trie.get(outputs[0].tolist() + [-1])[0]
        # print(corpus[c_id]['text'])
        # print(c_id)
        # print(c)
        total_num += 1
        for c_id in cid_list:
            if c_id in c:
                correct_num += 1

    print(correct_num, total_num, correct_num / total_num)

if __name__ == "__main__":
    main()