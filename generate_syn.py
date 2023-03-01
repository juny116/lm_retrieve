import logging
import random
import pickle
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
# from mteb import MTEB


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    random.seed(config['seed'])
    max_gen = config['max_gen']
    template = config['templates']['template']
    device = torch.device(config['device'])

    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    model = model.to(device)

    # MTEB version
    # dataset = MTEB(tasks=[config['task']]).tasks[0]
    # dataset.load_data(eval_splits=['test'])
    # rel = dataset.relevant_docs['test']
    # queries = dataset.queries['test']
    # corpus = dataset.corpus['test']

    dataset = config['task']
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = '/datasets/datasets/beir'
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    total_len = 0
    total_cnt = 0
    for k,v in corpus.items():
        total_len += len(tokenizer.encode(v['text']))
        total_cnt += 1
        if total_cnt == 100:
            break
    ave_len = int(total_len / total_cnt)
    
    results = []
    for i, (q_id, c) in enumerate(tqdm(qrels.items())):
        if i > max_gen:
            break

        c_id = next(iter(c))
        input_str = template.replace('[QUERY]', queries[q_id])

        input_ids = tokenizer.encode(input_str, return_tensors="pt")
        input_ids = input_ids.to(device)

        output = model.generate(input_ids,
            do_sample=config['generator']['do_sample'],                             
            max_new_tokens=ave_len+50, 
            top_k=config['generator']['top_k'], 
            top_p=config['generator']['top_p'], 
            min_length=ave_len-20,
            num_return_sequences=config['generator']['num_return_sequences']
        )
        try:
            result = {'q_id': q_id, 'c_id': c_id, 'query': queries[q_id], 'gt': corpus[c_id]['text'], 'output': [tokenizer.decode(output[j], skip_special_tokens=True) for j in range(int(config['generator']['num_return_sequences']))]}
            results.append(result)
        except Exception as e:
            print(f'Qid:{q_id} ERROR', e)
            

    result_dict = {result['q_id']: {'c_id': result['c_id'], 'query': result['query'], 'gt': result['gt'], 'output': result['output']} for result in results}

    p = Path(config['syn_doc_path'])
    p.mkdir(parents=True, exist_ok=True)
    with open(config['syn_doc_file'], 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    main()