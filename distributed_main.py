import logging
import os
import random
import time
import json

import evaluate
import hydra
from omegaconf import DictConfig
import torch
import deepspeed
import numpy as np
from datasets import load_dataset, load_metric, Dataset
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from tqdm.auto import tqdm
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    set_seed,
)
from trie import Trie
import pickle


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # To avoid warnings about parallelism in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    ds_config = dict(config['ds_configs'])

    # Process auto config
    # auto values are from huggingface
    if ds_config.get('zero_optimization'):
        if ds_config.get('zero_optimization').get('stage') == 3:
            if ds_config['zero_optimization']['reduce_bucket_size'] == 'auto':
                ds_config['zero_optimization']['reduce_bucket_size'] = config['generator']['hidden_size'] * config['generator']['hidden_size']
            if ds_config['zero_optimization']['stage3_prefetch_bucket_size'] == 'auto':
                ds_config['zero_optimization']['stage3_prefetch_bucket_size'] = config['generator']['hidden_size'] * config['generator']['hidden_size'] * 0.9
            if ds_config['zero_optimization']['stage3_param_persistence_threshold'] == 'auto':
                ds_config['zero_optimization']['stage3_param_persistence_threshold'] = config['generator']['hidden_size'] * 10

    # For huggingface deepspeed / Keep this alive!
    dschf = HfDeepSpeedConfig(ds_config)

    # Set logger
    logging.getLogger('beir').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    # logging level info for main process only
    logger.setLevel(logging.INFO)
    if local_rank > 0:
        logger.setLevel(logging.ERROR)

    if config['seed'] is not None:
        set_seed(config['seed'])
        random.seed(config['seed'])

    dataset = config['task']

    
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # corpus, queries, qrels = HFDataLoader(f"BeIR/{dataset}").load(split="test")

    if local_rank > 0:  
        logger.info("Waiting for main process to download datasets")
        torch.distributed.barrier()
    dataset = config['task']
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = '/datasets/datasets/beir'
    data_path = util.download_and_unzip(url, out_dir)
    if local_rank == 0:
        torch.distributed.barrier()

    ndcg = evaluate.load('ndcg.py', process_id=local_rank, num_process=world_size, experiment_id='ndcg')
    recall = evaluate.load('recall.py', process_id=local_rank, num_process=world_size, experiment_id='recall')
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    logger.info(f"Dataset loaded with {len(corpus)} documents and {len(queries)} queries and {len(qrels)} qrels")

    tokenizer = AutoTokenizer.from_pretrained(config['generator']["model_name_or_path"])

    if local_rank == 0:
        if config['create_trie']: 
            logger.info("Waiting for main process to create trie")
            sents = []
            for k, v in tqdm(corpus.items()):
                sents.append([0] + tokenizer.encode(v['text'], truncation=True, max_length=config['max_length']) + [-1, k])
            trie = Trie(sents)
            with open(f'results/{dataset}_{config["max_length"]}_trie.pkl', 'wb') as f:
                pickle.dump(trie.trie_dict, f)
            logger.info("Finish creating trie")
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            with open(f'results/{dataset}_{config["max_length"]}_trie.pkl', 'rb') as f:
                trie_dict = pickle.load(f)
                trie = Trie.load_from_dict(trie_dict)
    if local_rank > 0:
        torch.distributed.barrier()
        with open(f'results/{dataset}_{config["max_length"]}_trie.pkl', 'rb') as f:
            trie_dict = pickle.load(f)
            trie = Trie.load_from_dict(trie_dict)

    def prefix_allowed_fn(batch_id, sent):
        # print(batch_id, sent)
        sent = sent.tolist()
        trie_out = trie.get(sent)
        if trie_out == [-1]:
            trie_out = []
        return trie_out

    new_qrels = {'query_id': [], 'corpus_id': [], 'relevance': []}
    for k, v in qrels.items():
        new_qrels['query_id'].append(k)
        new_qrels['corpus_id'].append(list(v.keys()))
        new_qrels['relevance'].append(list(v.values()))

    hf_dataset = Dataset.from_dict(new_qrels)

    # Load model
    logger.info(f'Start loading model {config["generator"]["model_name_or_path"]}')
    model_loading_start = time.time()
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']["model_name_or_path"])

    model_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    model_engine.eval()

    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    sampler = torch.utils.data.distributed.DistributedSampler(hf_dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(hf_dataset, sampler=sampler, batch_size=1)

    template = config['templates']['template']
    progressbar = tqdm(range(len(dataloader)))

    errors = []
    for batch in dataloader:
        input_str = template.replace('[QUERY]', queries[batch['query_id'][0]])
        input_ids = tokenizer(input_str, truncation=True, max_length=510-config['max_length'], return_tensors="pt").input_ids.to(local_rank)

        with torch.no_grad():
            outputs = model_engine.module.generate(
                input_ids,
                max_new_tokens=config['max_length'],
                prefix_allowed_tokens_fn=prefix_allowed_fn,
                num_beams=10,
                num_return_sequences=10,
                remove_invalid_values=True,
            )

        cid_list = []
        for output in outputs:
            out_list = output.tolist()
            temp = [out_list[0]]
            for out in out_list[1:]:
                if out == 0:
                    break
                temp.append(out)
            try:
                cid_list.append(trie.get(temp + [-1])[0])
            except:
                errors.append(temp)
            

        # predictions = [1 for i in range(len(cid_list))]
        predictions = []
        for cid in cid_list:
            if cid in batch['corpus_id'][0]:
                predictions.append(1)
            else:
                predictions.append(0)

        ndcg.add(predictions=predictions)
        # recall.add(references=references, predictions=predictions)
        progressbar.update(1)

    print(errors)
    ndcg_results = ndcg.compute(k=[1,5,10])
    # recall_results = recall.compute(k=[1,5,10])
    if local_rank == 0:
        logger.info(ndcg_results)
        # logger.info(recall_results)

if __name__ == "__main__":
    main()
    
    