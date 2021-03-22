
import datetime
import pickle
import numpy as np
import torch
import time
import random
import multiprocessing as mp

import itertools
import os
from torch.utils.data import random_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DataProcessor, InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from transformers import get_linear_schedule_with_warmup



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def build_dataset_for_learning_first_sentence(tokenizer, path='data'):
    data_list = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))

    s1_list = []
    s2_list = []                            
    for instance in data_list:
        sorted_pair = sorted(zip(instance['indexes'], instance['sentences']))
        s1_list.append((sorted_pair[1][1], 1))
        for i in range(2, len(sorted_pair)):
            s2_list.append((sorted_pair[i][1], 0))
            
    s1_list += random.sample(s2_list, len(s1_list))
    random.shuffle(s1_list)
    
    input_ids = []
    attention_masks = []
    labels = []
    count = 0
    for sentence, label in s1_list:
        encoded_dict = tokenizer.encode_plus(
                        sentence,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_seq_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

        input_ids.append(encoded_dict['input_ids'])
        

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(label)
        if count % 100000 == 0:
            print(count)
        count += 1
 
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(
        input_ids, attention_masks, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    return train_dataset, val_dataset


    
def get_datasets_v2(tokenizer, max_seq_length):
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
    train_list = pickle.load(open('train_dataset_0.pkl', 'rb'))
    train_list = random.sample(train_list, 2000000)
    
    for count in range(len(train_list)):
        i, s1, s2, d, l = train_list[count]
        l = int(not l)
        encoded_dict = tokenizer.encode_plus(
                            s1,                      # Sentence to encode.
                            s2,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        token_type_ids.append(encoded_dict['token_type_ids'])

        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(l)
        if count % 100000 == 0:
            print(count)
 
    print("Finished?")
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    dataset = TensorDataset(
        input_ids, attention_masks, token_type_ids, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    return train_dataset, val_dataset


def get_datasets(tokenizer, max_seq_length):
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
    
    DATA_PATH = "data"
    data_list = pickle.load(open(os.path.join(DATA_PATH, 'train.pkl'), 'rb'))
    data_list = random.sample(data_list, 60000)
  
      
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
                            s1,                      # Sentence to encode.
                            s2,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        token_type_ids.append(encoded_dict['token_type_ids'])

        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(l)
        if count % 100000 == 0:
            print(count)
 
    print("Finished?")
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    dataset = TensorDataset(
        input_ids, attention_masks, token_type_ids, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    return train_dataset, val_dataset


if __name__=='__main__':
    print("hi")
    ts = time.time()

    max_seq_length = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     train_dataset, val_dataset =  build_dataset_for_learning_first_sentence(tokenizer)
    
    train_dataset, val_dataset = get_datasets(tokenizer, max_seq_length) 
    
    print(f'Time in parallel: {time.time() - ts}')

    with open("train_d_sample_6000.pkl", 'wb') as f:
        pickle.dump(train_dataset, f)

        
    with open("test_d_sample_6000.pkl", 'wb') as f:
        pickle.dump(val_dataset, f)

    