from sentence_transformers import SentenceTransformer
from mteb import MTEB, __version__
import logging
from time import time
import json
import os
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    syn_doc_path = config.get('syn_doc')
    task_name = config.get('task')

    with open(syn_doc_path, 'rb') as f:
        query_dict = pickle.load(f)

    model = SentenceTransformer(config['retriever']['name'])
    evaluation = MTEB(tasks=[task_name])

    eval_splits = None
    kwargs = {}
    evaluation_results = {}

    while len(evaluation.tasks) > 0:
        task = evaluation.tasks[0]
        task.save_suffix = 'ours'
        logger.info(f"\n\n********************** Evaluating {task.description['name']} **********************")

        if os.path.exists(config['save_path']) and config['overwrite_results'] is False:
            logger.warn(f"WARNING: {task.description['name']} results already exists. Skipping.")
            del evaluation.tasks[0]
            continue
        try:
            task_eval_splits = eval_splits if eval_splits is not None else task.description.get("eval_splits", [])

            # load data
            logger.info(f"Loading dataset for {task.description['name']}")
            task.load_data(eval_splits=task_eval_splits)

            if config['method'] != 'og':
                queries = task.queries['test']
                for key, value in queries.items():
                    queries[key] = query_dict[key]['output'][config['method']]

            # run evaluation
            task_results = {
                "mteb_version": __version__, 
                "dataset_revision": task.description.get("revision", None),
                "mteb_dataset_name": task.description['name'],
            }
            for split in task_eval_splits:
                tick = time()
                results = task.evaluate(model, split, target_devices=[0,1,2,4], **kwargs)
                tock = time()
                logger.info(f"Evaluation for {task.description['name']} on {split} took {tock - tick:.2f} seconds")
                results["evaluation_time"] = round(tock - tick, 2)
                task_results[split] = results
                logger.info(f"Scores: {results}")

            # save results
            with open(config['save_path'], "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)

            evaluation_results[task.description['name']] = task_results

        except Exception as e:
            logger.error(f"Error while evaluating {task.description['name']}: {e}")

        # empty memory
        del evaluation.tasks[0]

    print(evaluation_results)

if __name__ == "__main__":
    main()