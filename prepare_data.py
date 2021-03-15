import os
import pickle
import pandas as pd
import time
import multiprocessing as mp


def create_row(instance):
    output = []
    sorted_pair = sorted(zip(instance['indexes'], instance['sentences']))
    for i in range(len(sorted_pair)):
        for j in range(i+1, len(sorted_pair)):
            output.append([instance["ID"],  sorted_pair[i][1].lower(), sorted_pair[j][1].lower(), (j-i-1), 1])
            output.append([instance["ID"],  sorted_pair[j][1].lower(), sorted_pair[i][1].lower(), (j-i-1), 0])
    #print(f"len output {len(output)}")
    return output




def get_result(result):
    global results
    #print(results)
    results.extend(result)


if __name__ == '__main__':
   
    DATA_PATH = "data"

    train_list = pickle.load(open(os.path.join(DATA_PATH, 'train.pkl'), 'rb'))
    train_size = len(train_list) 
    
    results = []
    ts = time.time()
    pool = mp.Pool(mp.cpu_count())
    for i in range(train_size):
        pool.apply_async(create_row, args=(train_list[i], ), callback=get_result)
        if i % 100000 == 0:
            print(i)
    pool.close()
    pool.join()
    
    print(f'Time in parallel: {time.time() - ts}')
    print(len(results))
    with open('train_results.pkl', "wb") as f:
        pickle.dump(results, f)
