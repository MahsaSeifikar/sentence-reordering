import click
import pickle
import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
import itertools
import time
import datetime
import multiprocessing as mp


def main_second_sentence_v2():
    max_seq_length = 128

    first_model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_fist_sentence_extractor/'
    )
    second_model = BertForSequenceClassification.from_pretrained(
        './save_model/v2_d_sample_60000/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    first_model.cuda()
    first_model.eval()
    
    second_model.cuda()
    second_model.eval()
    
    device = "cuda"
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    cou = 0
    t = time.time()
    for test in test_list:
        # find fist sentence
        first_index = -1
        min_random = 1000
        less_random = -1
        max_score = -100
        max_index = -1
        for i, sentence in enumerate(test["sentences"]):
            encoded_dict = tokenizer.encode_plus(
                        sentence,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_seq_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
            
            b_input_ids = encoded_dict["input_ids"].to(device)
            b_input_mask = encoded_dict["attention_mask"].to(device)
            with torch.no_grad():        

                result = first_model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           return_dict=True)
                        
            logits = result.logits

            logits = logits.detach().cpu().numpy()
            
            if np.argmax(logits, axis=1).flatten():
                if max_score < logits[0][1]:
                    max_score = logits[0][1]
                    max_index = i
            else:
                if min_random > logits[0][0]:
                    min_random = logits[0][0]
                    less_random = i
                
            if max_score == -100:
                first_index = less_random
            else:
                first_index = max_index
        print(f"{first_index}")     
        cou += 1
        best_comb = 0
        best_score = -10000
        for combinations in list(itertools.permutations(range(6))):
            if combinations[0] != first_index:
                continue
            score_i = 0
            for i in range(len(combinations)-1):
                encoded_dict = tokenizer.encode_plus(
                    test["sentences"][combinations[i]], 
                    test["sentences"][combinations[i+1]], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                b_input_ids = encoded_dict["input_ids"].to(device)
                b_token_ids = encoded_dict["token_type_ids"].to(device)
                b_input_mask = encoded_dict["attention_mask"].to(device)
                with torch.no_grad():        

                    outputs = second_model(b_input_ids, 
                               token_type_ids=b_token_ids, 
                               attention_mask=b_input_mask, 
                               return_dict=True)
#                 outputs = second_model(**encoding)[0]
                outputs = outputs.logits.detach().cpu().numpy()
                
                score_i += outputs[0][1]
                
            if score_i > best_score:
                best_score = score_i
                best_comb = combinations
        
        
        if cou % 50 ==0 :        
            print(f"{cou} --> {format_time(time.time()-t)}")
        print(list(best_comb))
            
        results.append([test["ID"]]+list(best_comb))
        
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("d_result_score_v22H.csv", index=False)
    

def main_second_sentence_v1():
    max_seq_length = 128

#     first_model = BertForSequenceClassification.from_pretrained(
#         './save_model/v1_fist_sentence_extractor/'
#     )
    second_model = BertForSequenceClassification.from_pretrained(
        './save_model/v2_d_sample_60000/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
#     first_model.cuda()
#     first_model.eval()
    
    second_model.cuda()
    second_model.eval()
    
    device = "cuda"
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    cou = 0
    t = time.time()
    for test in test_list:
        # find fist sentence
        cou += 1
        best_comb = 0
        best_score = -10000
        for combinations in list(itertools.permutations(range(6))):
            score_i = 0
            for i in range(len(combinations)-1):
                encoded_dict = tokenizer.encode_plus(
                    test["sentences"][combinations[i]], 
                    test["sentences"][combinations[i+1]], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                b_input_ids = encoded_dict["input_ids"].to(device)
                b_token_ids = encoded_dict["token_type_ids"].to(device)
                b_input_mask = encoded_dict["attention_mask"].to(device)
                with torch.no_grad():        

                    outputs = second_model(b_input_ids, 
                               token_type_ids=b_token_ids, 
                               attention_mask=b_input_mask, 
                               return_dict=True)
#                 outputs = second_model(**encoding)[0]
                outputs = outputs.logits.detach().cpu().numpy()
                
                score_i += outputs[0][1]
                
            if score_i > best_score:
                best_score = score_i
                best_comb = combinations
    
        
        if cou % 10 ==0 :        
            print(f"{cou} --> {format_time(time.time()-t)}")
#         print(list(best_comb))
        results.append([test["ID"]]+list(best_comb))
        
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("d_result_score_v2_16h.csv", index=False)

def main_second_sentence_v0():
    max_seq_length = 128

    second_model = BertForSequenceClassification.from_pretrained(
        './save_model/v2_d_sample_60000/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    second_model.cuda()
    second_model.eval()
    
    device = "cuda"
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    cou = 0
    t = time.time()
    for test in test_list:
        cou += 1
        best_comb = []
        best_score = -10000
        matrix = []
        for i in range(6):
            temp = []
            for j in range(6):
                if i ==j:
                    temp.append(0)
                    continue
                encoded_dict = tokenizer.encode_plus(
                    test["sentences"][i], 
                    test["sentences"][j], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                b_input_ids = encoded_dict["input_ids"].to(device)
                b_token_ids = encoded_dict["token_type_ids"].to(device)
                b_input_mask = encoded_dict["attention_mask"].to(device)
                with torch.no_grad():        

                    outputs = second_model(b_input_ids, 
                               token_type_ids=b_token_ids, 
                               attention_mask=b_input_mask, 
                               return_dict=True)
                outputs = outputs.logits.detach().cpu().numpy()
                temp.append(outputs[0][1])
            matrix.append(temp)

        for combinations in list(itertools.permutations(range(6))):
            score_i = 0
            for i in range(len(combinations)-1):
                score_i += matrix[combinations[i]][combinations[i+1]]
                
            if score_i > best_score:
                best_score = score_i
                best_comb = combinations
    
        
        if cou % 50 ==0 :        
            print(f"{cou} --> {format_time(time.time()-t)}")
#         print(list(best_comb))
        results.append([test["ID"]]+list(best_comb))
        
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("d_result_score_v2_inteligence.csv", index=False)

        


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

if __name__=='__main__':
#     main_second_sentence_v1()
#     main_first_sentence()
    main_second_sentence_v0()
    