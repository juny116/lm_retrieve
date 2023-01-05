import pickle

base_path = 'generation'
file_name = "dbpedia"

with open(f'{base_path}/{file_name}.pkl', 'rb') as f:
    results = pickle.load(f)

result_dict = {result['q_id']: {'c_id': result['c_id'], 'query': result['query'], 'gt': result['gt'], 'output': result['output']} for result in results}

with open(f'{base_path}/{file_name}_dict.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
