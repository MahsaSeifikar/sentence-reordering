# ReorderSentence

This project aims to create a paragraph from 6 random sentences.

We have half a million dictionaries in train data, including a list of sentences and a list of the index that shows the correct order.

Due to lack of memory and GPU, a sampled data is created from the original dataset. We map the ordering sentences problem to a classifier with two possible classes. Either two sentences are continuous or not. For each paragraph, we consider the combination of 2 from 6 sentences which 5 of them have the label True, and the rest are False. Then, we finetune a BertForSequenceClassification from huggingface library. 

To order sentences in the test step, we consider all possible combinations of 6 sentences and, for all of them, calculate the sum of the probability of pair sentences, then choose the combination that leverages the max score.


## How to run?

To get the result, just run the following command.
```
chmod +x run.py
./run.py
```
