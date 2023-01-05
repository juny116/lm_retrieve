from sentence_transformers import SentenceTransformer
from mteb import MTEB, __version__
import logging
from time import time
import json
import os
import pickle
import torch

logger = logging.getLogger(__name__)

cuda = torch.device('cuda:7')

base_path = 'generation'
file_name = "dbpedia"

with open(f'{base_path}/{file_name}_dict.pkl', 'rb') as f:
    scifact = pickle.load(f)

# Define the sentence-transformers model name
model_name = "facebook/contriever-msmarco"

model = SentenceTransformer(model_name)
model = model.to(cuda)
evaluation = MTEB(tasks=["DBPedia"])

eval_splits = None
output_folder='results'
overwrite_results=True
kwargs = {}
evaluation_results = {}

while len(evaluation.tasks) > 0:
    task = evaluation.tasks[0]
    logger.info(f"\n\n********************** Evaluating {task.description['name']} **********************")

    if output_folder is not None:
        save_path = os.path.join(output_folder, f"{task.description['name']}{task.save_suffix}.json")
        if os.path.exists(save_path) and overwrite_results is False:
            logger.warn(f"WARNING: {task.description['name']} results already exists. Skipping.")
            del evaluation.tasks[0]
            continue
    try:
        task_eval_splits = eval_splits if eval_splits is not None else task.description.get("eval_splits", [])

        # load data
        logger.info(f"Loading dataset for {task.description['name']}")
        task.load_data(eval_splits=task_eval_splits)

        queries = task.queries['test']
        for key, value in queries.items():
            queries[key] = scifact[key]['output'][1]

        # run evaluation
        task_results = {
            "mteb_version": __version__, 
            "dataset_revision": task.description.get("revision", None),
            "mteb_dataset_name": task.description['name'],
        }
        for split in task_eval_splits:
            tick = time()
            results = task.evaluate(model, split, **kwargs)
            tock = time()
            logger.info(f"Evaluation for {task.description['name']} on {split} took {tock - tick:.2f} seconds")
            results["evaluation_time"] = round(tock - tick, 2)
            task_results[split] = results
            logger.info(f"Scores: {results}")

        # save results
        if output_folder is not None:
            with open(save_path, "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)

        evaluation_results[task.description['name']] = task_results

    except Exception as e:
        logger.error(f"Error while evaluating {task.description['name']}: {e}")


    # empty memory
    del evaluation.tasks[0]

print(evaluation_results)