import pickle
import multiprocessing as mp

from transformers import BertTokenizer, RobertaTokenizer

train_list = pickle.load(open('train_results.pkl', 'rb'))

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



def f(s1, s2):
    
    global max_len
    input_ids = tokenizer.encode(s1, s2, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))




max_len = 0
pool = mp.Pool(mp.cpu_count())
    
for i, s1, s2, _, l in train_list:

    pool.apply_async(f, args=(s1, s2, ))
    if i % 100000 == 0:
        print(i)
pool.close()
pool.join()


print('Max sentence length: ', max_len)

