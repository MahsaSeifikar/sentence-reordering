import click
import pickle
import torch
import os
import pandas as pd
import numpy as np
import itertools
import time

from transformers import BertForSequenceClassification, BertTokenizer
from utils import format_time

def evaluate_testset(test_list, model, tokenizer, num_sample):

    step = 0
    results = []
    t0 = time.time()
    for test in test_list:
        step+=1
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

    
        correct_output = []
        best_comb = list(best_comb)
        for i in range(6):
            correct_output.append(best_comb.index(i))
        
        results.append([test["ID"]]+correct_output)

        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            print(f'  Batch {step}  of  {len(test_list)}.    Elapsed: {elapsed}.')
            
        
    pd.DataFrame(results, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("d_result_score_v2_inteligence.csv", index=False)

        
@click.command()
@click.option('--max_seq_length', default=128, help="Maximum number of encoding sequence.")
@click.option('--num_sample', default=60000, help="Maximum number of sample from train data.")
def main(max_seq_length, num_sample):
    print("Prepare data is starting...)

    test_list = pickle.load(open(os.path.join('data', 'test.pkl'), 'rb'))
    model = BertForSequenceClassification.from_pretrained(
        f"./save_model/mode_{num_sample}"
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    model.cuda()
    model.eval()
    
    device = "cuda"
    evaluate_testset(test_list, model, tokenizer, num_sample)

if __name__=='__main__':
    main()
    