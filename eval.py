from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
from beir.retrieval import models
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import pickle

import os, random
import logging
import json


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
    # corpus, queries, qrels = HFDataLoader("BeIR/trec-covid").load(split="test")

    syn_doc = config.get('syn_doc_file')
    if os.path.exists(syn_doc):
        with open(syn_doc, 'rb') as f:
            query_dict = pickle.load(f)

    new_queries = {}

    if config['method'] == 'syn':
        print('syn')
        for key, value in qrels.items():
            new_queries[key] = query_dict[key]['output'][0]
    elif config['method'] == 'syn_q':
        print('syn_q')
        for key, value in qrels.items():
            new_queries[key] = query_dict[key]['output'][0] + value
    else:
        new_queries = queries

    queries = new_queries

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
    # SciFact is a relatively small dataset! (limit shards to 1)
    # number_of_shards = 1
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    # (2) For datasets with big corpus ==> keep default configuration
    if config['retriever']['type'] == 'sentence_transformers':
        model = DRES(models.SentenceBERT(config['retriever']['model_name_or_path']), batch_size=16, corpus_chunk_size=50000)
    elif config['retriever']['type'] == 'bm25':
        model = BM25(index_name=dataset, hostname='localhost', initialize=False)

        
    retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity

    if config['retriever']['type'] == 'sentence_transformers':
        if os.path.exists(config['corpus_embedding_file']):
            with open(config['corpus_embedding_file'], "rb") as f_in:
                pre_build_corpus_embeddings = pickle.load(f_in)
            # logger.warn(f"WARNING: {task.description['name']} results already exists. Skipping.")
            results, _ = retriever.retrieve(corpus, queries, corpus_embeddings=pre_build_corpus_embeddings)
        else:
            #### Retrieve dense results (format of results is identical to qrels)
            results, corpus_embeddings = retriever.retrieve(corpus, queries)
            corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

            # save results
            p = Path(config['corpus_embedding_path'])
            p.mkdir(parents=True, exist_ok=True)
            with open(config['corpus_embedding_file'], "wb") as f_out:
                pickle.dump(corpus_embeddings, f_out)
    else:
        results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    p = Path(config['save_path'])
    p.mkdir(parents=True, exist_ok=True)
    with open(config['save_file'], "w") as f_out:
        for metric in [ndcg, _map, recall, precision]:
            f_out.write(json.dumps(metric, ensure_ascii=False) + "\n")

    #### Retrieval Example ####
    # query_id, scores_dict = random.choice(list(results.items()))
    # logging.info("Query : %s\n" % queries[query_id])

    # scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    # for rank in range(10):
    #     doc_id = scores[rank][0]
    #     logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

if __name__ == "__main__":
    main()