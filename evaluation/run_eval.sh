#!/bin/bash

output_dir="/eval_output"

retrained_model_path=" " # Retrained model path  
base_model_path=" " # Unlearned model path
text_data_dir= " " #TO DO
data_split_path="./data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"
forget_set_path="./data_splits/forget_set_3per.csv"
synonym_file="./data_splits/Synonyms.csv" 

bert_config_path="bert_config.json"
bert_pretrained_dir="/allenai/scibert_scivocab_uncased_" # Path to pretrained BERT model

img_data_dir=" " # Path to folder with p10  p11  p12  etc
python eval_unlearning.py \
    --config "$bert_config_path" \
    --retrained_model_path "$retrained_model_path" \
    --base_model_path "$base_model_path" \
    --bert_pretrained_dir "$bert_pretrained_dir" \
    --output_dir "$output_dir" \
    --data_split_path "$data_split_path" \
    --forget_set_path "$forget_set_path" \
    --text_data_dir "$text_data_dir" \
    --synonym_file "$synonym_file" \
    --img_data_dir "$img_data_dir" \
    --batch_size 32 \
    --random_seed 42 \
    --do_eval 
