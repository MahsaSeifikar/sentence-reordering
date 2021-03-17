import click
import itertools
import os
import pickle
import time
import multiprocessing as mp


def create_train_data(instance):
    output = []
    sorted_pair = sorted(zip(instance['indexes'], instance['sentences']))
    for i in range(len(sorted_pair)):
        for j in range(i+1, len(sorted_pair)):
            output.append([instance["ID"],  sorted_pair[i][1].lower(), sorted_pair[j][1].lower(), (j-i-1), 1])
            output.append([instance["ID"],  sorted_pair[j][1].lower(), sorted_pair[i][1].lower(), (j-i-1), 0])
    return output


def create_test_data(instance):
    output = {}
    output["ID"] = instance["ID"]
    output["Pairs"] = []
    for a, b in itertools.combinations(stuff, 2):
        output["Pairs"].append([(a, b), (instance['sentences'][a], instance['sentences'][b])])
        output["Pairs"].append([(b, a), (instance['sentences'][b], instance['sentences'][a])])
        
    return output



def get_result(result):
    global results
    results.extend(result)


@click.command()
@click.option("--datatype",  default='test', help='It can be test or train')
@click.option('--path', default="data", help='data path')
def main(datatype, path):

    
    DATA_PATH = "data"
    if datatype == 'train':
        data_list = pickle.load(open(os.path.join(DATA_PATH, 'train.pkl'), 'rb'))
    else:
        data_list = pickle.load(open(os.path.join(DATA_PATH, 'test.pkl'), 'rb'))
    
    data_size = len(data_list) 
    
    results = []
    ts = time.time()
    pool = mp.Pool(mp.cpu_count())
    for i in range(data_size):
        if datatype == 'train':
            pool.apply_async(create_train_data, args=(train_list[i], ), callback=get_result)
        else
            pool.apply_async(create_test_data, args=(train_list[i], ), callback=get_result)
            
        if i % 100000 == 0:
            print(i)
    pool.close()
    pool.join()
    
    print(f'Time in parallel: {time.time() - ts}')
    print(len(results))
    
    with open(f'{datatype}_results.pkl', "wb") as f:
        pickle.dump(results, f)

        
if __name__ == '__main__':
    main()