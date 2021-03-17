
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


# input_ids = []
# token_type_ids = []
# attention_masks = []
# labels = []
# bigcount = 0
    
# def get_train_examples(s1, s2, l, max_seq_length):
# #     train_list = pickle.load(open('train_results.pkl', 'rb'))

# #     examples = []
# #     for i, s1, s2, _, l in train_list:
# #         examples.append(InputExample(guid=i, 
# #                                     text_a=s1, 
# #                                     text_b=s2, 
# #                                     label=l
# #                                     ))
# #     print("All train example processed...")
    
# #     return examples
#     global input_ids
#     global token_type_ids
#     global attention_masks
#     global labels
#     global bigcount
# #     bigcount += 1
# #     print("miay?")
#     encoded_dict = tokenizer.encode_plus(
#                             s1,                      # Sentence to encode.
#                             s2,
#                             add_special_tokens = True, # Add '[CLS]' and '[SEP]'
#                             max_length = max_seq_length,           # Pad & truncate all sentences.
#                             pad_to_max_length = True,
#                             return_attention_mask = True,   # Construct attn. masks.
#                             return_tensors = 'pt',     # Return pytorch tensors.
#                        )

#         # Add the encoded sentence to the list.    
#     input_ids.append(encoded_dict['input_ids'])
        
#     token_type_ids.append(encoded_dict['token_type_ids'])

#         # And its attention mask (simply differentiates padding from non-padding).
#     attention_masks.append(encoded_dict['attention_mask'])
#     labels.append(l)
    

    
def get_datasets(tokenizer, max_seq_length):
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    token_type_ids = []
    attention_masks = []
    labels = []
#     global input_ids
#     global token_type_ids
#     global attention_masks
#     global labels
#     global bigcount
    train_list = pickle.load(open('train_dataset_0.pkl', 'rb'))
#     pool = mp.Pool(mp.cpu_count()-5)
# 
    for count in range(len(train_list)-5000000):
        i, s1, s2, d, l = train_list[count]
        train_list[count] = None
#         if d >= 1:
#             continue
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

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(l)
#         pool.apply_async(get_train_examples, args=(s1, s2, l, max_seq_length, ))
        if count % 100000 == 0:
            print(count)
#             print(bigcount)
#         time.sleep(0.01)
#     pool.close()
#     pool.join()
 
    print("Finished?")
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
#     print('Original: ', sentences[0])
#     print('Token IDs:', input_ids[0])
    
#     label_list = [0, 1]
#     examples = get_train_examples()
#     features = convert_examples_to_features(examples,
#                                     tokenizer,
#                                     label_list=label_list,
#                                     max_length=max_seq_length,
#                                     output_mode= 'classification'
# #                                     pad_on_left=False,                 # pad on the left for xlnet
# #                                     pad_token=tokenizer.convert_tokens_to_ids(
# #                                         [tokenizer.pad_token])[0],
# #                                     pad_token_segment_id=0,
#                                     )
    
#     # Convert to Tensors and build dataset
#     all_input_ids = torch.tensor(
#         [f.input_ids for f in features], dtype=torch.long)
#     all_attention_mask = torch.tensor(
#         [f.attention_mask for f in features], dtype=torch.long)
#     all_token_type_ids = torch.tensor(
#         [f.token_type_ids for f in features], dtype=torch.long)

#     all_labels = torch.tensor(
#         [f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        input_ids, attention_masks, token_type_ids, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    return train_dataset, val_dataset




# def train_model(train_dataloader, validation_dataloader):

#     # This training code is based on the `run_glue.py` script here:
#     # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

#     # Set the seed value all over the place to make this reproducible.
#     seed_val = 42

#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     torch.cuda.manual_seed_all(seed_val)

#     # We'll store a number of quantities such as training and validation loss, 
#     # validation accuracy, and timings.
#     training_stats = []

#     # Measure the total training time for the whole run.
#     total_t0 = time.time()

#     # For each epoch...
#     for epoch_i in range(0, epochs):

#         # ========================================
#         #               Training
#         # ========================================

#         # Perform one full pass over the training set.

#         print("")
#         print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#         print('Training...')

#         # Measure how long the training epoch takes.
#         t0 = time.time()

#         # Reset the total loss for this epoch.
#         total_train_loss = 0

#         # Put the model into training mode. Don't be mislead--the call to 
#         # `train` just changes the *mode*, it doesn't *perform* the training.
#         # `dropout` and `batchnorm` layers behave differently during training
#         # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
#         model.train()

#         # For each batch of training data...
#         for step, batch in enumerate(train_dataloader):

#             # Progress update every 40 batches.
#             if step % 40 == 0 and not step == 0:
#                 # Calculate elapsed time in minutes.
#                 elapsed = format_time(time.time() - t0)

#                 # Report progress.
#                 print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

#             # Unpack this training batch from our dataloader. 
#             #
#             # As we unpack the batch, we'll also copy each tensor to the GPU using the 
#             # `to` method.
#             #
#             # `batch` contains three pytorch tensors:
#             #   [0]: input ids 
#             #   [1]: attention masks
#             #   [2]: labels 
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)

#             # Always clear any previously calculated gradients before performing a
#             # backward pass. PyTorch doesn't do this automatically because 
#             # accumulating the gradients is "convenient while training RNNs". 
#             # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
#             model.zero_grad()        

#             # Perform a forward pass (evaluate the model on this training batch).
#             # In PyTorch, calling `model` will in turn call the model's `forward` 
#             # function and pass down the arguments. The `forward` function is 
#             # documented here: 
#             # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
#             # The results are returned in a results object, documented here:
#             # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
#             # Specifically, we'll get the loss (because we provided labels) and the
#             # "logits"--the model outputs prior to activation.
#             result = model(b_input_ids, 
#                            token_type_ids=None, 
#                            attention_mask=b_input_mask, 
#                            labels=b_labels,
#                            return_dict=True)

#             loss = result.loss
#             logits = result.logits

#             # Accumulate the training loss over all of the batches so that we can
#             # calculate the average loss at the end. `loss` is a Tensor containing a
#             # single value; the `.item()` function just returns the Python value 
#             # from the tensor.
#             total_train_loss += loss.item()

#             # Perform a backward pass to calculate the gradients.
#             loss.backward()

#             # Clip the norm of the gradients to 1.0.
#             # This is to help prevent the "exploding gradients" problem.
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             # Update parameters and take a step using the computed gradient.
#             # The optimizer dictates the "update rule"--how the parameters are
#             # modified based on their gradients, the learning rate, etc.
#             optimizer.step()

#             # Update the learning rate.
#             scheduler.step()

#         # Calculate the average loss over all of the batches.
#         avg_train_loss = total_train_loss / len(train_dataloader)            

#         # Measure how long this epoch took.
#         training_time = format_time(time.time() - t0)

#         print("")
#         print("  Average training loss: {0:.2f}".format(avg_train_loss))
#         print("  Training epcoh took: {:}".format(training_time))

#         # ========================================
#         #               Validation
#         # ========================================
#         # After the completion of each training epoch, measure our performance on
#         # our validation set.

#         print("")
#         print("Running Validation...")

#         t0 = time.time()

#         # Put the model in evaluation mode--the dropout layers behave differently
#         # during evaluation.
#         model.eval()

#         # Tracking variables 
#         total_eval_accuracy = 0
#         total_eval_loss = 0
#         nb_eval_steps = 0

#         # Evaluate data for one epoch
#         for batch in validation_dataloader:

#             # Unpack this training batch from our dataloader. 
#             #
#             # As we unpack the batch, we'll also copy each tensor to the GPU using 
#             # the `to` method.
#             #
#             # `batch` contains three pytorch tensors:
#             #   [0]: input ids 
#             #   [1]: attention masks
#             #   [2]: labels 
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)

#             # Tell pytorch not to bother with constructing the compute graph during
#             # the forward pass, since this is only needed for backprop (training).
#             with torch.no_grad():        

#                 # Forward pass, calculate logit predictions.
#                 # token_type_ids is the same as the "segment ids", which 
#                 # differentiates sentence 1 and 2 in 2-sentence tasks.
#                 result = model(b_input_ids, 
#                                token_type_ids=None, 
#                                attention_mask=b_input_mask,
#                                labels=b_labels,
#                                return_dict=True)

#             # Get the loss and "logits" output by the model. The "logits" are the 
#             # output values prior to applying an activation function like the 
#             # softmax.
#             loss = result.loss
#             logits = result.logits

#             # Accumulate the validation loss.
#             total_eval_loss += loss.item()

#             # Move logits and labels to CPU
#             logits = logits.detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()

#             # Calculate the accuracy for this batch of test sentences, and
#             # accumulate it over all batches.
#             total_eval_accuracy += flat_accuracy(logits, label_ids)


#         # Report the final accuracy for this validation run.
#         avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#         print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

#         # Calculate the average loss over all of the batches.
#         avg_val_loss = total_eval_loss / len(validation_dataloader)

#         # Measure how long the validation run took.
#         validation_time = format_time(time.time() - t0)

#         print("  Validation Loss: {0:.2f}".format(avg_val_loss))
#         print("  Validation took: {:}".format(validation_time))

#         # Record all statistics from this epoch.
#         training_stats.append(
#             {
#                 'epoch': epoch_i + 1,
#                 'Training Loss': avg_train_loss,
#                 'Valid. Loss': avg_val_loss,
#                 'Valid. Accur.': avg_val_accuracy,
#                 'Training Time': training_time,
#                 'Validation Time': validation_time
#             }
#         )

#     print("")
#     print("Training complete!")

#     print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    

if __name__=='__main__':
    print("hi")
    ts = time.time()

    max_seq_length = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset, val_dataset = get_datasets(tokenizer, max_seq_length) 
    
    print(f'Time in parallel: {time.time() - ts}')

    with open("dummy_train_distance_0.pkl", 'wb') as f:
        pickle.dump(train_dataset, f)

        
    with open("dummy_val_distance_0.pkl", 'wb') as f:
        pickle.dump(val_dataset, f)

    
    # The DataLoader needs to know our batch size for training, so we specify it 
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
#     # size of 16 or 32.
#     batch_size = 32

#     # Create the DataLoaders for our training and validation sets.
#     # We'll take training samples in random order. 
#     train_dataloader = DataLoader(
#                 train_dataset,  # The training samples.
#                 sampler = RandomSampler(train_dataset), # Select batches randomly
#                 batch_size = batch_size # Trains with this batch size.
#             )

#     # For validation the order doesn't matter, so we'll just read them sequentially.
#     validation_dataloader = DataLoader(
#                 val_dataset, # The validation samples.
#                 sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
#                 batch_size = batch_size # Evaluate with this batch size.
#             )
    
#     model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
#     model.cuda()
    
#     # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
#     # I believe the 'W' stands for 'Weight Decay fix"
#     optimizer = AdamW(model.parameters(),
#                       lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                       eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                     )
    
#     # Number of training epochs. The BERT authors recommend between 2 and 4. 
#     # We chose to run for 4, but we'll see later that this may be over-fitting the
#     # training data.
#     epochs = 4

#     # Total number of training steps is [number of batches] x [number of epochs]. 
#     # (Note that this is not the same as the number of training samples).
#     total_steps = len(train_dataloader) * epochs

#     # Create the learning rate scheduler.
#     scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                                 num_warmup_steps = 0, # Default value in run_glue.py
#                                                 num_training_steps = total_steps)
