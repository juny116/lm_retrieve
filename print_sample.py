import os
import logging

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pickle


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

    syn_doc = config.get('syn_doc_file')
    if os.path.exists(syn_doc):
        with open(syn_doc, 'rb') as f:
            query_dict = pickle.load(f)

    for i, (key, value) in enumerate(qrels.items()):
        c_id = next(iter(value))
        print("---------------------")
        print('Query: ' + queries[key] + '\n')
        print('GT: '+ corpus[c_id]['text'] + '\n')
        print('Gen: '+ query_dict[key]['output'][0] + '\n')
        if i == 3:
            break

if __name__ == "__main__":
    main()