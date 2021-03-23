#!/bin/bash

echo Install Library
pip3 install -r requirements.txt
echo Libraries are here!

echo Preprocess data is starting...
python3 data_processing.py --num_sample 60000


echo Training model is starting...
python3 train.py  --num_sample 60000

# echo Evaluate the model is starting...
python3 evaluate.py --num_sample 60000