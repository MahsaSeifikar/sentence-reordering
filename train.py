import click
import datetime
import pickle
import numpy as np
import torch
import time
import random


from torch.utils.data import random_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from utils import format_time


def flat_accuracy(preds, labels):
    """ Calculate the accuracy of our predictions vs labels

    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def train_model(model, train_dataloader, validation_dataloader, epochs, scheduler, optimizer, num_sample):

    
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():    
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

       

        print("")
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')

        t0 = time.time()

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')
            
            
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
        print(f" Average training loss: {avg_train_loss}")
        print(f" Training epcoh took: {training_time}")

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
        print(f" Accuracy: {avg_val_accuracy}")

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print(f" Validation Loss: {avg_val_loss}")
        print("  Validation took: {validation_time}")

    print("")
    print("Training complete!")

    model.save_pretrained(f"save_model/mode_{num_sample}")

    


@click.command()
@click.option('--num_sample', default=60000, help="Maximum number of sample from train data.")
def main(num_sample):

    ts = time.time()

    
    with open(f"train_dataset_{num_sample}.pkl", 'rb') as f:
        train_dataset = pickle.load(f)

    with open(f"val_dataset_{num_sample}.pkl", 'rb') as f:
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
    train_model(model, train_dataloader, validation_dataloader, epochs, scheduler, optimizer, num_sample)


if __name__=='__main__':
    main()