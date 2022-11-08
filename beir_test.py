from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
from beir.retrieval.search.sparse import SparseSearch
from beir.retrieval import models

import pathlib, os, random
import logging
import json

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    print(args)
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    dataset = args.dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = '/datasets/datasets/beir'
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
    # SciFact is a relatively small dataset! (limit shards to 1)
    # number_of_shards = 1
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    # (2) For datasets with big corpus ==> keep default configuration
    model_name = args.model_name
    if args.model_type == 'SBERT':
        # model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
        model = DRES(models.SentenceBERT(args.model_path), batch_size=1)
    elif args.model_type == 'BM25':
        model = BM25(index_name=dataset, hostname='localhost', initialize=True)
    elif args.model_type == 'SPARTA':
        # model = SparseSearch(models.SPARTA("BeIR/sparta-msmarco-distilbert-base-v1"), batch_size=128)
        model = SparseSearch(models.SPARTA(args.model_path), batch_size=128)
    elif args.model_type == 'SPLADE':
        # model = DRES(models.SPLADE("naver/efficient-splade-V-large-doc"), batch_size=16)
        # model = DRES(models.SPLADE("naver/splade-cocondenser-selfdistil"), batch_size=16)
        model = DRES(models.SPLADE(args.model_path), batch_size=16)
        
    retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    with open(f'outputs/{dataset}_{model_name}.jsonl', "w", encoding='utf-8') as write_file:
        for metric in [ndcg, _map, recall, precision]:
            write_file.write(json.dumps(metric, ensure_ascii=False) + "\n")

    #### Retrieval Example ####
    # query_id, scores_dict = random.choice(list(results.items()))
    # logging.info("Query : %s\n" % queries[query_id])

    # scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    # for rank in range(10):
    #     doc_id = scores[rank][0]
    #     logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

if __name__ == "__main__":
    main()