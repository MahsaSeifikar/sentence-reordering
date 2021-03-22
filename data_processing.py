

import itertools
import os
import click
import datetime
import pickle
import numpy as np
import torch
import time
import random

from torch.utils.data import random_split
from torch.utils.data import TensorDataset

from utils import format_time


def get_tensor_dataset(tokenizer, max_seq_length, num_sample):
    """ Prepare sample of train data and split to train and validation.
    
    Args:
        tokenizer:
        max_seq_length:
        num_sample:

    Returns:
        (train_data, val_data): a tuple contain train and validation sample of original data

    """
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
    
    DATA_PATH = "data"
    data_list = pickle.load(open(os.path.join(DATA_PATH, 'train.pkl'), 'rb'))
    data_list = random.sample(data_list, num_sample)
  
      
    train_list = []
    for instance in data_list:
        sorted_pair = sorted(zip(instance['indexes'], instance['sentences']))
        for i, j in itertools.permutations(range(6), 2):
            if j-i == 1:
                for k in range(5):
                    train_list.append([instance["ID"],  sorted_pair[i][1].lower(), sorted_pair[j][1].lower(), 1])
            else:
                train_list.append([instance["ID"],  sorted_pair[i][1].lower(), sorted_pair[j][1].lower(), 0])
        
    
    for count in range(len(train_list)):
        i, s1, s2, l = train_list[count]
        encoded_dict = tokenizer.encode_plus(
                            s1,                     
                            s2,
                            add_special_tokens = True, 
                            max_length = max_seq_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                       )

        input_ids.append(encoded_dict['input_ids'])
        
        token_type_ids.append(encoded_dict['token_type_ids'])

        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(l)

        if count % 100000 == 0:
            print(f'{count} of Train data are encoded!')
 
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    dataset = TensorDataset(
        input_ids, attention_masks, token_type_ids, labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f'{train_size} training samples')
    print(f'{val_size} validation samples')
    
    return train_dataset, val_dataset

@click.command()
@click.option('--max_seq_length', default=128, help="Maximum number of encoding sequence.")
@click.option('--num_sample', default=60000, help="Maximum number of sample from train data.")
def main(max_seq_length, num_sample):
    print("Prepare data is starting....")
    ts = time.time()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset, val_dataset = get_tensor_dataset(tokenizer, max_seq_length, num_sample) 
    
    print(f'Time in parallel: {time.time() - ts}')

    with open(f"train_dataset_{num_sample}.pkl", 'wb') as f:
        pickle.dump(train_dataset, f)
        
    with open(f"val__dataset_{num_sample}.pkl", 'wb') as f:
        pickle.dump(val_dataset, f)

if __name__=='__main__':
    main()    
    