
import datetime
import pickle
import numpy as np
import torch
import time
import random
import multiprocessing as mp


from torch.utils.data import random_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers import DataProcessor, InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertForNextSentencePrediction

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



def train_model(model, train_dataloader, validation_dataloader, epochs, scheduler, optimizer):

    
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():    

        device = torch.device("cuda")
    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    training_stats = []

    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

       

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            model.zero_grad()        
            result = model(b_input_ids, 
                           token_type_ids=b_token_type_ids, 
                           attention_mask=b_input_mask, 
                           labels=b_labels,
                           return_dict=True)

            loss = result.loss
            logits = result.logits

            total_train_loss += loss.item()

            loss.backward()

            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

           
            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        
        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
     
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_token_type_ids = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            with torch.no_grad():        

                result = model(b_input_ids, 
                           token_type_ids=b_token_type_ids, 
                           attention_mask=b_input_mask, 
                           labels=b_labels,
                           return_dict=True)

            loss = result.loss
            logits = result.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    model.save_pretrained("save_model/v2_d_sample_60000")

    

if __name__=='__main__':
    print("start...")
    ts = time.time()

    max_seq_length = 128
    
    with open("train_d_sample_6000.pkl", 'rb') as f:
        train_dataset = pickle.load(f)

    with open("test_d_sample_6000.pkl", 'rb') as f:
        val_dataset = pickle.load(f)

    
    batch_size = 32

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
    )
#     model = BertForNextSentencePrediction.from_pretrained(
#         'bert-base-uncased',
#     )
    model.cuda()
    
 
    optimizer = AdamW(model.parameters(),
                      lr = 1e-5, 
                      eps = 1e-8 
                    )
    
    
    epochs = 1

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    
    # Train Model
    train_model(model, train_dataloader, validation_dataloader, epochs, scheduler, optimizer)