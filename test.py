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

def main_second_sentence():
    max_seq_length = 128

    first_model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_fist_sentence_extractor/'
    )
    second_model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_second_sentence_extractor/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    first_model.cuda()
    first_model.eval()
    
    second_model.cuda()
    second_model.eval()
    
    device = "cuda"
    next_model = BertForNextSentencePrediction.from_pretrained(
        'bert-base-uncased'
    )
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    c = 0
    k = 0
    f = 0
    for test in test_list:
        k += 1
        
        temp_r = [test["ID"]]
        visited = [0] * 6
        current_node = None
        min_random = 1000
        less_random = -1
        max_score = -100
        max_index = -1
        second_pos = []
        # find fist sentence
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
                second_result = second_model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask, 
                           return_dict=True)
                        
            second_pos.append(second_result.logits.detach().cpu().numpy())
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
            visited[less_random] = 1
            current_node = less_random
            f+=1
        else:
            visited[max_index] = 1
            current_node = max_index
        
        temp_r.append(current_node)
                
        count = 1
        print("Find first sentence")
        while not all(visited):
            change = 0
            min_random = 1000
            less_random = -1
            max_score = -100
            max_index = -1
            for j in np.nonzero(np.array(visited)==0)[0]:
                encoding = tokenizer.encode_plus(
                    test["sentences"][current_node], 
                    test["sentences"][j], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                outputs = next_model(**encoding)[0]
                outputs = outputs.detach().cpu().numpy()
                if count ==1 and np.argmax(second_pos[j], axis=1).flatten() == 1:
                    visited[j] = 1
                    current_node = j
                    temp_r.append(current_node)
                    change = 1
                    count +=1
                    break
                    
                if np.argmax(outputs, axis=1).flatten() == 0:
                    visited[j] = 1
                    current_node = j
                    temp_r.append(current_node)
                    change = 1
                    break
                    
#                     if max_score < outputs[0][0]:
#                         max_score = outputs[0][0]
                        
#                         max_index = j
#                         change = 1
                else:
                    if min_random > outputs[0][1]:
                        min_random = outputs[0][1]
                        less_random = j
                    
            if change == 0:
                visited[less_random] = 1
                current_node = less_random
                temp_r.append(less_random)
                
                c+=1
#             else: 
#                 visited[max_index] = 1
#                 current_node = max_index
#                 temp_r.append(max_index)
        
        results.append(temp_r)
        
        print(f"k: {k} f:{f} c:{c}")
    
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("results_first_sentence_2.csv", index=False)
    

def main_first_sentence():
    max_seq_length = 128

    model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_fist_sentence_extractor/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.cuda()
    model.eval()
    device = "cuda"
    next_model = BertForNextSentencePrediction.from_pretrained(
        'bert-base-uncased'
    )
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    c = 0
    k = 0
    f = 0
    for test in test_list:
        k += 1
        
        temp_r = [test["ID"]]
        visited = [0] * 6
        current_node = None
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

                result = model(b_input_ids, 
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
            visited[less_random] = 1
            current_node = less_random
            f+=1
        else:
            visited[max_index] = 1
            current_node = max_index
        
        temp_r.append(current_node)
                
                
        print("Find first sentence")
        while not all(visited):
            change = 0
            min_random = 1000
            less_random = -1
            max_score = -100
            max_index = -1
            for j in np.nonzero(np.array(visited)==0)[0]:
                encoding = tokenizer.encode_plus(
                    test["sentences"][current_node], 
                    test["sentences"][j], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                outputs = next_model(**encoding)[0]
                outputs = outputs.detach().cpu().numpy()
                if np.argmax(outputs, axis=1).flatten() == 0:
                    visited[j] = 1
                    current_node = j
                    temp_r.append(current_node)
                    change = 1
                    break
                    
#                     if max_score < outputs[0][0]:
#                         max_score = outputs[0][0]
                        
#                         max_index = j
#                         change = 1
                else:
                    if min_random > outputs[0][1]:
                        min_random = outputs[0][1]
                        less_random = j
                    
            if change == 0:
                visited[less_random] = 1
                current_node = less_random
                temp_r.append(less_random)
                
                c+=1
#             else: 
#                 visited[max_index] = 1
#                 current_node = max_index
#                 temp_r.append(max_index)
        
        results.append(temp_r)
        
        print(f"k: {k} f:{f} c:{c}")
    
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("results_firs_second_sentence_1.csv", index=False)
    
          
        
def main_find_max_numbert():
    max_seq_length = 128

    model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_fist_sentence_extractor/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.cuda()
    model.eval()
    device = "cuda"
    next_model = BertForNextSentencePrediction.from_pretrained(
        'bert-base-uncased'
    )
                
    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
    results = []
    for test in test_list:
        k += 1
        
        temp_r = [test["ID"]]
        visited = [0] * 6
        current_node = None
        min_random = 1000
        less_random = -1
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

                result = model(b_input_ids, 
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
            visited[less_random] = 1
            current_node = less_random
            f+=1
        else:
            visited[max_index] = 1
            current_node = max_index
        
        temp_r.append(current_node)
                
                

        while not all(visited):
            change = 0
            min_random = 1000
            less_random = -1
            max_score = -100
            max_index = -1
            for j in np.nonzero(np.array(visited)==0)[0]:
                encoding = tokenizer.encode_plus(
                    test["sentences"][current_node], 
                    test["sentences"][j], 
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = max_seq_length,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_tensors = 'pt',     # Return pytorch tensors.

                )
                outputs = next_model(**encoding)[0]
                outputs = outputs.detach().cpu().numpy()
                if np.argmax(outputs, axis=1).flatten() == 0:
                    visited[j] = 1
                    current_node = j
                    temp_r.append(current_node)
                    change = 1
                    break
                    
#                     if max_score < outputs[0][0]:
#                         max_score = outputs[0][0]
                        
#                         max_index = j
#                         change = 1
                else:
                    if min_random > outputs[0][1]:
                        min_random = outputs[0][1]
                        less_random = j
                    
            if change == 0:
                visited[less_random] = 1
                current_node = less_random
                temp_r.append(less_random)
                
                c+=1
#             else: 
#                 visited[max_index] = 1
#                 current_node = max_index
#                 temp_r.append(max_index)
        
        results.append(temp_r)
        
        print(f"k: {k} f:{f} c:{c}")
    
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("results_firs_second_sentence_1.csv", index=False)



   

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
        break
                
        print(cou)
#         print(list(best_comb))
        results.append([test["ID"]]+list(best_comb))
        
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("d_result_score_v1.csv", index=False)



@click.command()
@click.option('--path', default='data/adjacency_matrix', help='Path to save results.')
def main(path):
    max_seq_length = 128
    
    with open('test_results.pkl', 'rb') as f:
        test_list = pickle.load(f)
    
    model = BertForSequenceClassification.from_pretrained(
        './save_model/v1_distance_0/'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.cuda()
    model.eval()
    device = "cuda"
    count = 0
    for instance in test_list:
        adjacency_list = []
        count+=1
        for i, j, s1, s2 in instance['Pairs']:
            
            encoded_dict = tokenizer.encode_plus(
                            s1,                      # Sentence to encode.
                            s2,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
           
            b_input_ids = encoded_dict["input_ids"].to(device)
            b_input_mask = encoded_dict["attention_mask"].to(device)
            b_token_type_ids = encoded_dict["token_type_ids"].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                           token_type_ids=b_token_type_ids, 
                           attention_mask=b_input_mask, 
                           return_dict=True)
            
            logits = result.logits

            logits = logits.detach().cpu().numpy()
            adjacency_list.append([i, j, logits, s1, s2])
            
        with open(os.path.join(path, str(instance['ID'])+".pkl"), 'wb') as f:
            pickle.dump(adjacency_list, f)
            
#         if count % 100==0:
#             print(count)
    
        

if __name__=='__main__':
#     main()
#     main_first_sentence()
    main_second_sentence_v1()
#     test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    
#     results = []
#     for test in test_list:
        
#         temp_r = [test["ID"]]
#         for i, sentence in enumerate(test["sentences"]):
#             temp_r.append(sentence)
        
#         results.append(temp_r)
        
#     pd.DataFrame(results, columns= ["ID", "s1", "s2", "s3", "s4", "s5", "s6"]).to_csv("test_pandas.csv", index=False)
        
    