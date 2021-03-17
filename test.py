import click
import pickle
import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer



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
            
        if count % 100==0:
            print(count)
    
        

if __name__=='__main__':
    main()
    