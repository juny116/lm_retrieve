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

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(len(corpus), len(queries), len(qrels))
    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    model = model.to(device)


    if config['chunk_corpus']:
        trie_path = f'results/{dataset}_{config["max_length"]}_chunk_trie.pkl'
    else:
        trie_path = f'results/{dataset}_{config["max_length"]}_trie.pkl'
    if config['create_trie']:
        sents = []
        for k, v in tqdm(corpus.items()):
            if config['chunk_corpus']:
                tokenized_input = tokenizer.encode(v['text'], max_length=2048)
                num_chunks = len(tokenized_input) // config['max_length']
                for i in range(num_chunks):
                    sents.append([0] + tokenized_input[i*config['max_length']:(i+1)*config['max_length']] + [-1, k])
            else:
                sents.append([0] + tokenizer.encode(v['text'], truncation=True, max_length=config['max_length']) + [-1, k])
        if config['chunk_corpus']:
            print(f'num chunks: {num_chunks}')
        print(f'num sents: {len(sents)}')
        trie = Trie(sents)
        with open(trie_path, 'wb') as f:
            pickle.dump(trie.trie_dict, f)
    else:
        with open(trie_path, 'rb') as f:
            trie_dict = pickle.load(f)
        trie = Trie.load_from_dict(trie_dict)

    def prefix_allowed_fn(batch_id, sent):
        # print(batch_id, sent)
        sent = sent.tolist()
        trie_out = trie.get(sent)
        if len(trie_out) > 1 and -1 in trie_out:
            print(trie_out)
            return [0]
        if trie_out == [-1]:
            # print(trie_out)
            trie_out = [0]
        return trie_out

    template = config['templates']['template']
    print(template)

    ndcg = evaluate.load('metric/ndcg.py', experiment_id='ndcg')
    recall = evaluate.load('metric/recall.py', experiment_id='recall')

    total_num = 0
    correct_num = 0
    errors = []
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        if i > max_gen:
            break
        input_str = template.replace('[QUERY]', queries[q_id])
        input_ids = tokenizer(input_str, return_tensors="pt", max_length=2048, truncation=True).input_ids.to(device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=config['max_length']+1,
            prefix_allowed_tokens_fn=prefix_allowed_fn,
            num_beams=config['num_beams'],
            num_return_sequences=10,
            remove_invalid_values=True,
            use_cache=True,
        )

        cid_list = []
        for output in outputs:
            # print(output.tolist())
            # print(tokenizer.decode(output.tolist()))
            out_list = output.tolist()
            temp = [out_list[0]]
            for out in out_list[1:]:
                if out == 0:
                    break
                temp.append(out)
            try:
                cid_list.append(trie.get(temp + [-1])[0])
                if trie.get(temp + [-1])[0] == []:
                    print(q_id, c, temp)
                    return
                # else:
                #     print(q_id, c, trie.get(temp + [-1])[0])
            except:
                errors.append(temp)
        
        total_num += 1
        for c_id in cid_list:
            if c_id in c:
                correct_num += 1

        predictions = [1 for i in range(len(cid_list))]
        references = []
        for cid in cid_list:
            if cid in c:
                references.append(1)
            else:
                references.append(0)

        ndcg.add(references=references, predictions=predictions)
        recall.add(references=references, predictions=predictions)

    print(correct_num, total_num, correct_num / total_num)
    print(errors)
    ndcg_results = ndcg.compute(k=[1,5,10])
    recall_results = recall.compute(k=[1,5,10])
    print(ndcg_results)
    print(recall_results)
    p = Path(config['save_path'])
    p.mkdir(parents=True, exist_ok=True)
    with open(config['save_file'], "w") as f_out:
        for metric in [ndcg_results, recall_results]:
            f_out.write(json.dumps(metric, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()