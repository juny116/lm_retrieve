import os, random, sys
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
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from time import time
import evaluate
from accelerate import init_empty_weights, infer_auto_device_map
import numpy as np
import tracemalloc
from utils import display_top, convert_unit, SIZE_UNIT


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    tracemalloc.start()
    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    dataset = config["task"]
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    out_dir = "/datasets/datasets/beir"
    data_path = util.download_and_unzip(url, out_dir)
    device = torch.device(config["device"])
    max_gen = config["max_gen"]

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(len(corpus), len(queries), len(qrels))

    if config["generator"]["name"] == "llama2-13b-chat":
        model_name = "meta-llama/Llama-2-13b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="sequential",
            max_memory={0: "40GiB"},
            # max_memory={0: "20GiB", 1: "40GiB", 2: "40GiB", 3: "40GiB", 4: "40GiB"},
        ).eval()

        # with torch.inference_mode():
        #     tokenized_inputs = tokenizer(msgs, return_tensors='pt', padding=True, truncation=True, return_token_type_ids=False).to(model.device)
        #     output_tokens = model.generate(**tokenized_inputs, **{'do_sample':False})
        #     outputs_generated_only = output_tokens[:,tokenized_inputs["input_ids"].shape[-1]:]

        #     output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True, spaces_between_special_tokens=False)

    elif config["generator"]["name"] == "flan-ul2":
        model_id = config["generator"]["model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_config = T5Config.from_pretrained(model_id)

        max_memory = {i: "48GiB" for i in range(8)}  # Assume 4 GPUs
        max_memory[0] = "20GiB"  # to fit lm_head to the same device as the inputs

        with init_empty_weights():
            model = T5ForConditionalGeneration(model_config)
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=["T5Block"],
                dtype=torch.float16,
                max_memory=max_memory,
            )
        device_map["lm_head"] = device_map["decoder.embed_tokens"]
        model = T5ForConditionalGeneration.from_pretrained(
            model_id, device_map=device_map, torch_dtype=torch.float16
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["generator"]["model_name_or_path"]
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["generator"]["model_name_or_path"]
        )
        model = model.to(device)

    if config["chunk_corpus"]:
        trie_path = f'results/{dataset}_{config["max_length"]}_{config["generator"]["name"]}_chunk_trie.pkl'
    else:
        trie_path = f'results/{dataset}_{config["max_length"]}_{config["generator"]["name"]}_trie.pkl'
    if config["create_trie"]:
        sents = []
        for k, v in tqdm(corpus.items()):
            if config["chunk_corpus"]:
                tokenized_input = tokenizer.encode(
                    v["text"], max_length=2048, truncation=True
                )
                num_chunks = len(tokenized_input) // config["max_length"]
                for i in range(num_chunks):
                    sents.append(
                        tokenized_input[
                            i * config["max_length"] : (i + 1) * config["max_length"]
                        ]
                        + [-1, k]
                    )
            else:
                sents.append(
                    tokenizer.encode(
                        v["text"], truncation=True, max_length=config["max_length"]
                    )
                    + [-1, k]
                )
        print(f"num sents: {len(sents)}")
        trie = Trie(sents)
        with open(trie_path, "wb") as f:
            pickle.dump(trie.trie_dict, f)
    else:
        print("Loading trie")
        with open(trie_path, "rb") as f:
            trie_dict = pickle.load(f)
        print("Loaded trie")
        trie = Trie.load_from_dict(trie_dict)
        print("Loaded trie from dict")

    def prefix_allowed_fn(batch_id, sent):
        sent = sent.tolist()
        trie_out = trie.get(sent[input_len:])
        if len(trie_out) > 1 and -1 in trie_out:
            print(trie_out)
            return [tokenizer.eos_token_id]
        if trie_out == [-1]:
            trie_out = [tokenizer.eos_token_id]
        return trie_out

    template = config["templates"]["template"]

    errors = []
    results = {}
    total_set = set()
    start = True
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        results[q_id] = {}
        if i >= max_gen:
            break
        input_str = template.replace("[QUERY]", queries[q_id])
        with torch.inference_mode():
            input_ids = tokenizer(
                input_str, return_tensors="pt", max_length=2048, truncation=True
            ).input_ids.to(device)
            input_len = input_ids.shape[1]
            outputs = model.generate(
                input_ids,
                max_new_tokens=config["max_length"] + 1,
                prefix_allowed_tokens_fn=prefix_allowed_fn,
                num_beams=config["num_beams"],
                num_return_sequences=config["num_beams"],
                remove_invalid_values=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        for output, score in zip(
            outputs.sequences, torch.exp(outputs.sequences_scores)
        ):
            out_list = output.tolist()
            temp = []
            for out in out_list[input_len:]:
                if out == tokenizer.eos_token_id:
                    break
                temp.append(out)
            try:
                retrieved = trie.get(temp + [-1])
                total_set.update(retrieved)
                for cid in retrieved:
                    if cid not in results[q_id]:
                        # converted_score = np.float32(score.item())
                        converted_score = score.item()
                        results[q_id][cid] = converted_score
                        if start:
                            print(type(converted_score))
                            print(converted_score)
                            start = False
            except:
                errors.append(temp)

        # for cid in corpus:
        #     if cid not in results[q_id]:
        #         results[q_id][cid] = 0

    print("TRACKING MEMORY")
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, limit=20)
    print("TRACKING MEMORY")

    retriever = EvaluateRetrieval()
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values, ignore_identical_ids=False
    )
    print(f"ndcg: {ndcg}")
    print(f"map: {_map}")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print(f"total set: {len(total_set)}")
    print(f"corpus: {len(corpus)}")
    p = Path(config["save_path"])
    p.mkdir(parents=True, exist_ok=True)
    with open(config["save_file"], "w") as f_out:
        for metric in [ndcg, _map, recall, precision]:
            f_out.write(json.dumps(metric, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    main()
