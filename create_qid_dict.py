import pickle

base_path = 'generation'
file_name = "nq_4"

with open(f'{base_path}/{file_name}.pkl', 'rb') as f:
    results = pickle.load(f)

result_dict = {result['q_id']: {'c_id': result['c_id'], 'query': result['query'], 'gt': result['gt'], 'output': result['output']} for result in results}
for result in results:
    print("---------------------")
    print('Query: ' + result['query'] + '\n')
    print('GT: '+ result['gt'] + '\n')

    print('1: '+ result['output'][0])
    # print('2: '+ result['output'][1])
    # print('3: '+ result['output'][2])

# with open(f'{base_path}/{file_name}_dict.pkl', 'wb') as f:
#     pickle.dump(result_dict, f)
