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
from datasets import load_dataset, load_metric
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    start_time = time.time()

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
    metric = evaluate.load('ndcg.py', process_id=local_rank, num_process=world_size)
    
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    corpus, queries, qrels = HFDataLoader(f"BeIR/{dataset}").load(split="test")


    logger.info(f"Dataset loaded with {len(corpus)} documents and {len(queries)} queries and {len(qrels)} qrels")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load model
    logger.info(f'Start loading model {config["generator"]["model_name_or_path"]}')
    model_loading_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config['generator']["model_name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']["model_name_or_path"])

    model_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    model_engine.eval()

    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    # text_in = [queries[i]['text'] for i in range(local_rank * 20, 20 * (local_rank+1))]
    text_in = [queries[i]['text'] for i in range(0 * 20, 20 * (0+1))]
    print(len(text_in))
    # if local_rank == 0:
    #     text_in = ["Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy", 
    #                "Is this review positive or negative? Review: this is the worst restaurant ever"]
    # elif local_rank == 1:
    #     text_in = ["Is this review positive or negative? Review: this is the worst restaurant ever", "This is test"]
    # elif local_rank == 2:
    #     text_in = ["Is this review positive or negative?", "This is test 2"]
    # elif local_rank == 3:
    #     text_in = ["Is this review", "this is test 333"]

    # inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)

    inputs = tokenizer(text_in, padding=True, return_tensors="pt").to(device=local_rank)

    start_gen = time.time()
    with torch.no_grad():
        outputs = model_engine.module.generate(
            inputs=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=384,
            num_beams=10,
            num_return_sequences=10,
            use_cache=True,
            remove_invalid_values=True,
            synced_gpus=True
        )
    end_gen = time.time()
    logger.info(f'Generation time {end_gen - start_gen}')
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"rank{local_rank}:\n   in={text_in[0]}\n  out={text_out}")

    # # Preprocessing the datasets
    # def preprocess_function(examples):
    #     # Tokenize the texts
    #     result = {}
    #     texts = examples[config['datasets']['sentence1_key']] if config['datasets']['sentence2_key'] is None else examples[config['datasets']['sentence1_key']] + '\n' + examples[config['datasets']['sentence2_key']]
        
    #     result['inputs'] = texts
    #     # result = tokenizer(*texts, padding=config['padding'], max_length=config['max_length'], truncation=True)
    #     result["labels"] = examples["label"]

    #     return result

    # # Ensures the main process performs the mapping
    # if local_rank > 0:  
    #     logger.info("Waiting for main process to perform the mapping")
    #     torch.distributed.barrier()
    # # processed_datasets = raw_datasets.map(
    # #     preprocess_function,
    # #     remove_columns=raw_datasets["train"].column_names,
    # #     desc="Running tokenizer on dataset",
    # # )
    # if local_rank == 0:
    #     torch.distributed.barrier()

    # batch_size = ds_config['train_micro_batch_size_per_gpu']

    # # Evaluate! 
    # logger.info("***** Few-shot Evaluation *****")
    # logger.info(f"  TASK                                = {config['datasets']['task']}")
    # logger.info(f"  Num TRAIN examples                  = {len(train_dataset)}")
    # logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
    # logger.info(f"  Random Seed                         = {config['seed']}")
    # logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    # metric = load_metric('accuracy', num_process=world_size, process_id=local_rank)

    # for epoch in range(config['epochs']):
    #     model_engine.module.eval()
    #     progressbar = tqdm(range(len(test_dataloader)))
    #     for step, batch in enumerate(test_dataloader):
    #         inputs = tokenizer(batch['inputs'], padding=config['padding'], max_length=config['max_length'], return_tensors='pt').to(device=local_rank)
    #         inputs['labels'] = torch.Tensor(batch['labels']).to(device=local_rank)
    #         with torch.no_grad():
    #             loss, predictions = model_engine(**inputs)

    #         metric.add_batch(predictions=predictions, references=batch['labels'])
    #         progressbar.update(1)

    #     result = metric.compute()
    #     if local_rank == 0:
    #         logger.info(f"Epoch {epoch}: Evaluation accuracy {result['accuracy'] * 100}")
    #     #save checkpoint
    #     model_engine.save_checkpoint(config['save_path'], epoch)
   
    # end_time = time.time()
    # logger.info(f'Total runtime : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    main()
    
    