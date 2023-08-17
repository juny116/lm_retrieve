from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import random


def main() -> None:
    random.seed(42)
    dataset = "nq"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    out_dir = "/datasets/datasets/beir"
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    print(len(corpus), len(queries), len(qrels))

    # randomly sample 10 queries and related corpus documents, and print them
    for i in range(10):
        qid = random.choice(list(queries.keys()))
        print("Query:", queries[qid])
        print("Related docs:", [corpus[docid] for docid in qrels[qid]])
        # print number of related docs
        print("Related docs:", len(qrels[qid]))
        print()
    # calculate average number of related docs per query
    print(
        "Average number of related docs per query:",
        sum([len(qrels[qid]) for qid in qrels]) / len(qrels),
    )
    # calculate average length of documents
    print(
        "Average length of documents:",
        sum([len(corpus[docid]["text"].split()) for docid in corpus]) / len(corpus),
    )


if __name__ == "__main__":
    main()
