from mteb.tasks import SciFact, NQ, MSMARCOv2, MSMARCO, DBPedia, FEVER, QuoraRetrieval
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from mteb import MTEB
from pathlib import Path
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    random.seed(config['seed'])
    max_gen = config['max_gen']
    template = config['templates']['template']
    device = torch.device(config['device'])

    tokenizer = AutoTokenizer.from_pretrained(config['generator']['model_name_or_path'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['generator']['model_name_or_path'])
    model = model.to(device)

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
            max_length=config['generator']['max_length'], 
            top_k=config['generator']['top_k'], 
            top_p=config['generator']['top_p'], 
            min_length=config['generator']['min_length'],
            num_return_sequences=config['generator']['num_return_sequences']
        )

        result = {'q_id': q_id, 'c_id': c_id, 'query': queries[q_id], 'gt': corpus[c_id]['text'], 'output': [tokenizer.decode(output[j], skip_special_tokens=True) for j in range(3)]}
        results.append(result)

    result_dict = {result['q_id']: {'c_id': result['c_id'], 'query': result['query'], 'gt': result['gt'], 'output': result['output']} for result in results}

    p = Path(config['syn_doc_path'])
    p.mkdir(parents=True, exist_ok=True)
    with open(config['syn_doc_file'], 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    main()