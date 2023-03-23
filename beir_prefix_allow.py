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
from trie import Trie
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config
from tqdm import tqdm
from time import time
import evaluate
from accelerate import init_empty_weights, infer_auto_device_map
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
    device = torch.device(config['device'])
    max_gen = config['max_gen']

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    print(len(corpus), len(queries), len(qrels))
    if config['generator']['name'] == 'flan-ul2':
        model_id = config['generator']['model_name_or_path']
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_config = T5Config.from_pretrained(model_id)

        max_memory={i: "48GiB" for i in range(8)}  # Assume 4 GPUs
        max_memory[0] = "20GiB"  # to fit lm_head to the same device as the inputs

        with init_empty_weights():
            model = T5ForConditionalGeneration(model_config)
            device_map = infer_auto_device_map(model, no_split_module_classes=["T5Block"], dtype=torch.float16, max_memory=max_memory)
        device_map['lm_head'] = device_map["decoder.embed_tokens"]
        model = T5ForConditionalGeneration.from_pretrained(model_id, device_map=device_map, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
        model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
        model = model.to(device)

    if config['chunk_corpus']:
        trie_path = f'results/{dataset}_{config["max_length"]}_{config["generator"]["name"]}_chunk_trie.pkl'
    else:
        trie_path = f'results/{dataset}_{config["max_length"]}_{config["generator"]["name"]}_trie.pkl'
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
            trie_out = [0]
        return trie_out

    template = config['templates']['template']
    print(template)

    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  # +2 as we start our idx at 0
        return dcg
    # ndcg_old = evaluate.load('metric/ndcg.py', experiment_id='ndcg')
    # ndcg = {"ndcg_at_" + str(k): [] for k in [5]}
    errors = []
    results = {}
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        results[q_id] = {}
        if i > max_gen:
            break
        input_str = template.replace('[QUERY]', queries[q_id])
        input_ids = tokenizer(input_str, return_tensors="pt", max_length=2048, truncation=True).input_ids.to(device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=config['max_length']+1,
            prefix_allowed_tokens_fn=prefix_allowed_fn,
            num_beams=config['num_beams'],
            num_return_sequences=config['num_beams'],
            remove_invalid_values=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        for output, score in zip(outputs.sequences, torch.exp(outputs.sequences_scores)):
            out_list = output.tolist()
            temp = [out_list[0]]
            for out in out_list[1:]:
                if out == 0:
                    break
                temp.append(out)
            try:
                retrieved = trie.get(temp + [-1])
                for cid in retrieved:
                    if cid not in results[q_id]:
                        results[q_id][cid] = score.item()
                        # results[q_id][cid] = 1
            except:
                errors.append(temp)
        # print(results[q_id])

        # # NDCG@k
        # for k_val in [5]:
        #     predicted_relevance = [
        #         1 if top_hit in c else 0 for top_hit in list(results[q_id].keys())[0:k_val]
        #     ]
        #     true_relevances = [1] * len(c)
        #     dcg = compute_dcg_at_k(predicted_relevance, k_val)
        #     print(dcg)
        #     idcg = compute_dcg_at_k(true_relevances, k_val)
        #     print(idcg)
        #     ndcg_value = dcg / idcg 
        #     print(ndcg_value)
        #     ndcg["ndcg_at_" + str(k_val)].append(ndcg_value)
        #     true_relevances = ([1] * len(c) + [0] * (k_val - len(c)))[:k_val]
        #     ndcg_old.add(predictions=predicted_relevance, reference=true_relevances)

        for cid in corpus:
            if cid not in results[q_id]:
                results[q_id][cid] = 0
    # for k in ndcg:
    #     ndcg[k] = np.mean(ndcg[k])
    # print(ndcg)
    # ndcg_results = ndcg_old.compute(k=[5])
    # print(ndcg_results)
    retriever = EvaluateRetrieval()
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


    print(errors)
    
    p = Path(config['save_path'])
    p.mkdir(parents=True, exist_ok=True)
    with open(config['save_file'], "w") as f_out:
        for metric in [ndcg, _map, recall, precision]:
            f_out.write(json.dumps(metric, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()