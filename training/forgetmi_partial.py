#!/usr/bin/env python3
from datetime import datetime
import os
import random
import pandas as pd
from tqdm import tqdm, trange
from scipy.stats import logistic
from scipy.special import softmax
import logging
import numpy as np
import json
import sklearn
import time
import csv
from torch.utils.data import DataLoader, Dataset, Sampler
import copy
import wandb
import time
import re
import csv
from sklearn.model_selection import train_test_split
import random

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torch.optim.lr_scheduler as lr_scheduler

from evaluation.eval_unlearning import get_probability_measure
import joint_img_txt.metrics as eval_metrics
from joint_img_txt import main_utils, parser
from joint_embedding import Gate, Outer, Attention


from transformers import BertTokenizer
from transformers.optimization import AdamW

import joint_img_txt.loss as custom_loss
from joint_img_txt.model_utils import CXRImageTextDataset, EdemaClassificationProcessor, RandomTranslateCrop, CenterCrop, EdemaMultiLabelClassificationProcessor
from joint_img_txt.model import ImageTextModel
from joint_img_txt.convert_examples_to_features import convert_examples_to_features, convert_examples_to_features_multilabel
from sklearn.model_selection import train_test_split

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten_metrics(metrics, prefix=""):
    """
    Flattens nested metric dictionaries, appending index for list values.
    """
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Recursively flatten dictionaries
            flat_metrics.update(flatten_metrics(value, prefix=f"{prefix}{key}."))
        elif isinstance(value, list):
            # Log each list element separately
            for i, v in enumerate(value):
                flat_metrics[f"{prefix}{key}_{i}"] = v
        else:
            flat_metrics[f"{prefix}{key}"] = value
    return flat_metrics

class AlignedSampler(Sampler):
    def __init__(self, dataset_length, shuffle=False, seed=None):
        self.dataset_length = dataset_length
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(self.dataset_length))
        if self.shuffle:
            if self.seed is not None:
                torch.manual_seed(self.seed) 
            torch.randperm(len(indices))  # Shuffle the indices
        return iter(indices)

    def __len__(self):
        return self.dataset_length

def euclidean_distance(embed1, embed2):
        return torch.sqrt(torch.sum((embed1 - embed2)**2, dim=1))

def cosine_similarity_loss(embed1, embed2):
    norm1 = torch.norm(embed1, dim=1)
    norm2 = torch.norm(embed2, dim=1)
    return torch.sum(embed1 * embed2, dim=1) / (norm1 * norm2)

def build_dataset(args, tokenizer, image_noise_params=None):
    logger = logging.getLogger(__name__)

    '''
    Load text features if they have been pre-processed;
    otherwise pre-process the raw text and save the features
    '''
    processor = EdemaMultiLabelClassificationProcessor() \
        if args.output_channel_encoding == 'multilabel' \
        else EdemaClassificationProcessor()
    num_labels = len(processor.get_labels())

    if args.output_channel_encoding == 'multilabel':
        get_features = convert_examples_to_features_multilabel
    else:
        get_features = convert_examples_to_features
    cached_features_file = os.path.join(
        args.text_data_dir,
        f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")
    cached_noisy_features_file = os.path.join(
        args.text_data_dir,
        f"cachednoisyfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")
    if os.path.exists(cached_features_file) and os.path.exists(cached_noisy_features_file) and not args.reprocess_input_data:
        print(args.reprocess_input_data)
        logger.info("Loading features from cached file %s", cached_features_file)
        print("Loading features from cached file %s"%cached_features_file)
        features = torch.load(cached_features_file)
        noisy_features = torch.load(cached_noisy_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.text_data_dir)
        label_list = processor.get_labels()

        synonyms_df = pd.read_csv(args.synonyms_dir)
        text_noise_level = args.text_noise_level
        examples = processor.get_all_examples(args.text_data_dir)
        noisy_examples = processor.get_noisy_examples(args.text_data_dir, synonyms_df, text_noise_level)

        features = get_features(examples, label_list, args.max_seq_length, tokenizer)
        noisy_features = get_features(noisy_examples, label_list, args.max_seq_length, tokenizer)

        logger.info("Saving features into cached file %s", cached_features_file)
        print("Saving features into cached file %s"%cached_features_file)
        torch.save(features, cached_features_file)
        torch.save(noisy_features, cached_noisy_features_file)

    all_txt_tokens = {f.report_id: f.input_ids for f in features}
    all_txt_masks = {f.report_id: f.input_mask for f in features}
    all_txt_segments = {f.report_id: f.segment_ids for f in features}
    all_txt_labels = {f.report_id: f.label_id for f in features}

    noisy_txt_tokens = {f.report_id: f.input_ids for f in noisy_features}
    noisy_txt_masks = {f.report_id: f.input_mask for f in noisy_features}
    noisy_txt_segments = {f.report_id: f.segment_ids for f in noisy_features}
    noisy_txt_labels = {f.report_id: f.label_id for f in noisy_features}

    retain_img_labels, retain_img_txt_ids, val_img_labels, val_img_txt_ids, test_img_labels, test_img_txt_ids, rand_img_labels, rand_img_txt_ids, \
        forget_img_labels, forget_img_txt_ids, n_retain, n_val, n_test, n_rand, n_forget = data_split(args.data_split_path, args.forget_set_path, args.random_point_ratio, args.validation_ratio)

    '''
    Specify the image pre-processing method 
    depending on it's for training/evaluation
    '''
    if args.do_train:
        xray_transform = RandomTranslateCrop(2048)
    if args.do_eval:
        xray_transform = CenterCrop(2048)

    '''
    Instantiate the image-text dataset
    '''
    retain_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, retain_img_txt_ids, args.img_data_dir, 
                                  retain_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform,
                                  output_channel_encoding = args.output_channel_encoding)

    test_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, test_img_txt_ids, args.img_data_dir, 
                                  test_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    rand_dataset = CXRImageTextDataset(args.id, 
                                  noisy_txt_tokens, noisy_txt_masks, noisy_txt_segments, 
                                  noisy_txt_labels, rand_img_txt_ids, args.img_data_dir, 
                                  rand_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, perturb_img=True,
                                  noise_params=image_noise_params, output_channel_encoding = args.output_channel_encoding)

    forget_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, forget_img_txt_ids, args.img_data_dir, 
                                  forget_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    val_dataset = CXRImageTextDataset(args.id, 
                                all_txt_tokens, all_txt_masks, all_txt_segments, 
                                all_txt_labels, val_img_txt_ids, args.img_data_dir, 
                                val_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                output_channel_encoding=args.output_channel_encoding)
                      
    print("Length of the retaining dataset is ", len(retain_dataset))
    print("Length of the random dataset is ", len(rand_dataset))
    print("Length of the forget dataset is ", len(forget_dataset))

    dataset = {
        'retain': retain_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'random': rand_dataset,
        'forget': forget_dataset,
        'n_retain': n_retain,
        'n_val': n_val,  
        'n_test': n_test,
        'n_rand': n_rand,
        'n_forget': n_forget,
    }

    return dataset, num_labels
def data_split(split_list_path, forget_ids_path, rand_ratio, validation_ratio=0.1):

    random.seed(0)

    print('Data split list being used: ', split_list_path)
    print('Forget list being used: ', forget_ids_path)

    train_labels = {}
    train_img_txt_ids = {}
    test_labels = {}
    test_img_txt_ids = {}
    rand_labels = {}
    rand_img_txt_ids = {}
    forget_labels = {}
    forget_img_txt_ids = {}

    with open(split_list_path, 'r') as train_label_file:
        forget_ids = pd.read_csv(forget_ids_path)
        forget_ids = forget_ids.astype(str)
        forget_ids = forget_ids.subject_id.values

        train_label_file_reader = csv.reader(train_label_file)
        header = next(train_label_file_reader)
        print("Header skipped:", header)

        for row in train_label_file_reader:
            if row == header or row[3] == 'edeme_severity':
                print(f"Skipping repeated header or invalid row: {row}")
                continue
            try:
                severity = float(row[3])  # Convert severity to float
            except ValueError:
                print(f"Skipping non-numeric row: {row}")
                continue  # Skip invalid rows
            if row[0] in forget_ids:
                forget_labels[row[2]] = [severity]
                forget_img_txt_ids[row[2]] = row[1]

                rand_labels[row[2]] = [severity]
                rand_img_txt_ids[row[2]] = row[1]
            else:
                if row[-1] == 'TEST':
                    test_labels[row[2]] = [severity]
                    test_img_txt_ids[row[2]] = row[1]
                else:
                    train_labels[row[2]] = [severity]
                    train_img_txt_ids[row[2]] = row[1]

    # VALIDATION DATASET
    train_ids = list(train_img_txt_ids.keys())
    train_values = list(train_img_txt_ids.values())
    train_labels_list = list(train_labels.values())

    train_ids, val_ids, train_values, val_values, train_labels_split, val_labels_split = train_test_split(
        train_ids, train_values, train_labels_list, test_size=validation_ratio, random_state=42, stratify=train_labels_list)

    # Add validation labels and images back into training set
    train_labels.update(dict(zip(val_ids, val_labels_split)))
    train_img_txt_ids.update(dict(zip(val_ids, val_values)))

    val_labels = dict(zip(val_ids, val_labels_split))
    val_img_txt_ids = dict(zip(val_ids, val_values))

    train_labels.update(val_labels)
    train_img_txt_ids.update(val_img_txt_ids)

    n_train, n_val, n_rand, n_test, n_forget = len(train_img_txt_ids), len(val_img_txt_ids), len(rand_img_txt_ids), len(test_img_txt_ids), len(forget_img_txt_ids)

    print("Total number of training labels: ", len(train_labels))
    print("Total number of training DICOM IDs: ", len(train_img_txt_ids))
    print("Total number of validation labels: ", len(val_labels))
    print("Total number of validation DICOM IDs: ", len(val_img_txt_ids))
    print("Total number of testing labels: ", len(test_labels))
    print("Total number of testing DICOM IDs: ", len(test_img_txt_ids))
    print("Total number of unlearning labels: ", len(forget_labels))
    print("Total number of unlearning DICOM IDs: ", len(forget_img_txt_ids))
    print("Total number of random labels: ", len(rand_labels))
    print("Total number of random DICOM IDs: ", len(rand_img_txt_ids))

    return train_labels, train_img_txt_ids, val_labels, val_img_txt_ids, test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids, forget_labels, forget_img_txt_ids, n_train, n_val, n_test, n_rand, n_forget

def get_model_inputs(args, dataset, device):

    image, label_raw, txt_ids, txt_mask, txt_segment_ids, label_onehot_or_ordinal, report_id = dataset

    image = image.to(device)
    label_raw = label_raw.to(device)
    txt_ids = txt_ids.to(device)
    txt_mask = txt_mask.to(device)
    txt_segment_ids = txt_segment_ids.to(device)
    label_onehot_or_ordinal = label_onehot_or_ordinal.to(device)
    report_id = report_id.to(device)

    inputs = {  
                'input_img':                image,
                'input_ids':                txt_ids,
                'attention_mask':           txt_mask,
                'token_type_ids':           txt_segment_ids,
                'labels':                   None,
                'bert_pool_last_hidden':    args.bert_pool_last_hidden,
                'bert_pool_use_img':        args.bert_pool_use_img,
                'bert_pool_img_lowerlevel': args.bert_pool_img_lowerlevel
            } 
    return inputs, label_raw, report_id

def unlearn(args, output_dir, device, model_og, model_ul, model_re, optimizer, optimizer_grouped_parameters, scheduler, tokenizer, dataset, num_labels, alpha, beta, theta, gamma):
    retain_set, val_set, rand_set, forget_set, test_set = (
        dataset['retain'], dataset['validation'], dataset['random'], dataset['forget'], dataset['test']
    )    
    n_retain, n_val, n_rand, n_forget, n_test = (
        dataset['n_retain'], dataset['n_val'], dataset['n_rand'], dataset['n_forget'], dataset['n_test']
    )

    aligned_sampler = AlignedSampler(len(forget_set), shuffle=True, seed=42)

    print('Retrieving forget set data of length ', len(forget_set))
    forget_dataloader = DataLoader(forget_set, sampler=aligned_sampler, 
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers, 
                                  pin_memory=True)

    print('Retrieving validation set data of length ', len(val_set))
    val_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), 
                                batch_size=args.eval_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)

    print('Retrieving random set data of length ', len(rand_set))
    rand_dataloader = DataLoader(rand_set, sampler=aligned_sampler, 
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers, 
                                  pin_memory=True)
    
    print('Retrieving test set data of length ', len(test_set))
    test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set),
                                  batch_size=args.eval_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)

    print('Starting the Unlearning Process...')
    n_epochs = args.unlearn_epochs
    unlearning_iterator = trange(int(n_epochs), desc="Epoch")

    model_re.eval()
    model_ul.train()
    model_og.train()

    unlearning_start_time = time.time()

    for epoch in unlearning_iterator:
        # ------------------------------------------- UNLEARNING ------------------------------------------- 
        total_loss = 0
        md_loss, uu_loss = 0, 0
        mkr_loss, ukr_loss = 0, 0

        print('Retrieving retain set data of length ', len(retain_set))
        retain_sampler = RandomSampler(retain_set)
        retain_dataloader = DataLoader(retain_set, sampler=retain_sampler, batch_size=args.unlearn_batch_size,
                                    num_workers=args.num_cpu_workers, pin_memory=True)

        epoch_iterator = tqdm(zip(forget_dataloader, rand_dataloader, retain_dataloader), desc="Retain Set Iteration")

        steps = 0
        for (forget_batch, rand_batch, retain_batch) in epoch_iterator:
            # ------------------------------------------- GET INPUTS FROM ORIGINAL AND UNLEARNING MODELS -------------------------------------------
            retain_batch = tuple(t.to(device=device, non_blocking=True) for t in retain_batch)
            retain_inputs, retain_labels, retain_report_id = get_model_inputs(args, retain_batch, device)
            original_retain_outputs = model_og(**retain_inputs)
            og_ret_img_emb, og_ret_img_log, og_ret_txt_emb, og_ret_txt_log = original_retain_outputs[:4]
            unlearn_retain_outputs = model_ul(**retain_inputs)
            ul_ret_img_emb, ul_ret_img_log, ul_ret_txt_emb, ul_ret_txt_log = unlearn_retain_outputs[:4]
            
            rand_batch = tuple(t.to(device=device, non_blocking=True) for t in rand_batch)
            rand_inputs, rand_labels, rand_report_id = get_model_inputs(args, rand_batch, device)
            original_rand_outputs = model_og(**rand_inputs)
            unlearn_rand_outputs = model_ul(**rand_inputs)
            og_rand_img_emb, og_rand_img_log, og_rand_txt_emb, og_rand_txt_log = original_rand_outputs[:4]
            ul_rand_img_emb, ul_rand_img_log, ul_rand_txt_emb, ul_rand_txt_log = unlearn_rand_outputs[:4]
            
            forget_batch = tuple(t.to(device=device, non_blocking=True) for t in forget_batch)
            forget_inputs, forget_labels, forget_report_id = get_model_inputs(args, forget_batch, device)
            original_forget_outputs = model_og(**forget_inputs)
            unlearn_forget_outputs = model_ul(**forget_inputs)
            ul_frgt_img_emb, ul_frgt_img_log, ul_frgt_txt_emb, ul_frgt_txt_log = unlearn_forget_outputs[:4]
            og_frgt_img_emb, og_frgt_img_log, og_frgt_txt_emb, og_frgt_txt_log = original_forget_outputs[:4]

            # ------------------------------------------- JOINT EMBEDDINGS -------------------------------------------
            gate_ul_ret =  Gate(inp1_size = ul_ret_img_emb.shape[1], inp2_size = ul_ret_txt_emb.shape[1]).to(device)
            ul_ret_joint_emb = gate_ul_ret(ul_ret_img_emb, ul_ret_txt_emb)

            gate_ul_frgt =  Gate(inp1_size = ul_frgt_img_emb.shape[1], inp2_size = ul_frgt_txt_emb.shape[1]).to(device)
            ul_frgt_joint_emb = gate_ul_frgt(ul_frgt_img_emb, ul_frgt_txt_emb)

            gate_og_frgt =  Gate(inp1_size = og_frgt_img_emb.shape[1], inp2_size = og_frgt_img_emb.shape[1]).to(device)
            og_frgt_joint_emb = gate_ul_frgt(og_frgt_img_emb, og_frgt_img_emb)

            gate_og_rand =  Gate(inp1_size = og_rand_img_emb.shape[1], inp2_size = og_rand_txt_emb.shape[1]).to(device)
            og_rand_joint_emb = gate_og_rand(og_rand_img_emb, og_rand_txt_emb)

            gate_og_ret =  Gate(inp1_size = og_ret_img_emb.shape[1], inp2_size = og_ret_txt_emb.shape[1]).to(device)
            og_ret_joint_emb = gate_og_ret(og_ret_img_emb, og_ret_txt_emb)

            # we use large noise therefore its minimization problem
            if args.use_noise:
                # ------------------------------------------- UU Loss -------------------------------------------
                ul_frgt_concat_emb = torch.cat((ul_frgt_img_emb, ul_frgt_txt_emb), dim=-1)
                og_rand_concat_emb = torch.cat((og_rand_img_emb, og_rand_txt_emb), dim=-1)

                L_uu = (euclidean_distance(ul_frgt_concat_emb, og_rand_concat_emb).mean())

                # ------------------------------------------- MD Loss -------------------------------------------
                L_md = euclidean_distance(ul_frgt_joint_emb, og_rand_joint_emb).mean()
            # don't use any noise, maximization problem
            else:
                # ------------------------------------------- UU Loss -------------------------------------------
                ul_frgt_concat_emb = torch.cat((ul_frgt_img_emb, ul_frgt_txt_emb), dim=-1)
                og_rand_concat_emb = torch.cat((og_rand_img_emb, og_rand_txt_emb), dim=-1)

                L_uu = (euclidean_distance(ul_frgt_concat_emb, og_rand_concat_emb).mean())
                L_uu = -L_uu
                # ------------------------------------------- MD Loss -------------------------------------------
                L_md = euclidean_distance(ul_frgt_joint_emb, og_rand_joint_emb).mean()
                L_md = -L_md

            # ------------------------------------------- UKR Loss -------------------------------------------
            ul_ret_concat_emb = torch.cat((ul_ret_img_emb, ul_ret_txt_emb), dim=-1)
            og_ret_concat_emb = torch.cat((og_ret_img_emb, og_ret_txt_emb), dim=-1)

            L_ukr = euclidean_distance(ul_ret_concat_emb, og_ret_concat_emb).mean()
    
            # ------------------------------------------- MKR Loss -------------------------------------------
            L_mkr = euclidean_distance(ul_ret_joint_emb, og_ret_joint_emb).mean()

            # ------------------------------------------- Hinge Loss -------------------------------------------
            if epoch == 0:
                margin_ukr = (L_ukr + 1).detach() 
                margin_mkr = (L_mkr + 1).detach()

            if epoch != 0:
                L_ukr = torch.minimum(L_ukr, margin_ukr)
                L_mkr = torch.minimum(L_mkr, margin_mkr)

            # ------------------------------------------- Total Loss -------------------------------------------
            loss = (alpha*L_ukr + beta*L_uu) + (theta*L_md + gamma*L_mkr)

            #------------------------------------------- Backpropagation -------------------------------------------
            md_loss += L_md.item()
            uu_loss += L_uu.item()
            mkr_loss += L_mkr.item()
            ukr_loss += L_ukr.item()
            total_loss += loss.item()
            
            if epoch != 0:
                loss.backward()

                if 'grad' in optimizer_grouped_parameters:
                    torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, 
                                                args.max_grad_norm)

            steps += 1

        if epoch != 0:
            optimizer.step()
            optimizer.zero_grad()  

        # ------------------------------------------- Unlearning evaluation -------------------------------------------
        # metrics, cached_metrics = evaluate(
        #     model_og, model_ul, model_re, 
        #     retain_dataloader, forget_dataloader, val_dataloader, test_dataloader,  
        #     device, args, cached_metrics, mode='val')  
            
        # flattened_metrics = flatten_metrics(metrics)
        cosine_similarity = get_probability_measure(
            args,
            copy.deepcopy(model_re).eval(),
            copy.deepcopy(model_ul).eval(),
            retain_dataloader,
            device
        )

        wandb.log({
            "Epoch": epoch,
            "MD Loss": md_loss / steps,
            "UU Loss": uu_loss / steps,
            "MKR Loss": mkr_loss / steps,
            "UKR Loss": ukr_loss / steps,
            "Total Loss": total_loss / steps,
            "Learning Rate": args.learning_rate,
            "Cosine Similarity": cosine_similarity
            # **flattened_metrics
        })

        print(f'Total Loss After epoch {epoch} = {total_loss/steps}')
        print(f"UKR: {ukr_loss / steps}, UU: {uu_loss / steps}, MD: {md_loss / steps}, MKR: {mkr_loss / steps}")
        
        # ------------------------------------------- Save Unlearning Model -------------------------------------------
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        model_to_save = model_ul.module if hasattr(model_ul, 'module') else model_ul
        # model_to_save.save_pretrained(epoch_output_dir)
        torch.save(model_to_save.state_dict(), os.path.join(epoch_output_dir, "model_state_dict.pth"))


    # ------------------------------------------- Log Unlearning Time Taken -------------------------------------------    
    unlearning_end_time = time.time()  
    unlearning_duration = (unlearning_end_time - unlearning_start_time) / 3600

    wandb.log({
        "time(hours)": unlearning_duration
    })
    
    return retain_dataloader, forget_dataloader, val_dataloader, test_dataloader
    
def main():
    # ------------------------------------------- WANDB & LOSS COEFFICIENTS & NOISE ------------------------------------------- 
    wandb.init(project="unlearning-mbzuai", entity="forget_exp")
    config = wandb.config

    alpha = config.alpha
    beta = config.beta
    theta = config.theta
    gamma = config.gamma

    total = alpha + beta + theta + gamma

    alpha = round(alpha / total, 2)
    beta = round(beta / total, 2)
    theta = round(theta / total, 2)
    gamma = round(gamma / total, 2)

    image_noise_params = {"mean": config.noise_mean, "std": config.noise_std}
    unlearning_percentage_match = re.search(r'forget_set_(\d+)per\.csv', config.forget_set_path)
    unlearning_percentage = int(unlearning_percentage_match.group(1)) if unlearning_percentage_match else None

    wandb.log({
        "seed": config.random_seed,
        "unlearning_percentage": unlearning_percentage,
        "use_noise": config.use_noise,
        "val_ratio": config.validation_ratio,
        "random_point_ratio": config.random_point_ratio,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "unlearn_epochs": config.unlearn_epochs,
        "scheduler": config.scheduler,
        "image_noise_mean": config.noise_mean,
        "image_noise_std": config.noise_std,
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        "gamma": gamma,
    })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{alpha}*UKR_{beta}*UU_t{theta}*MD_{gamma}*MKR_mean{image_noise_params['mean']}_std{image_noise_params['std']}_{config.use_noise}_{timestamp}_partials_hinge_euc"
    wandb.run.name = run_name

    # ------------------------------------------- SEED & DEVICE ------------------------------------------- 
    set_seed(config.random_seed)
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # ------------------------------------------- LOAD MODELS ------------------------------------------- 
    output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Unlearned model will be saved in {output_dir}")

    model_og = ImageTextModel.from_pretrained(config.base_model_path).to(device)
    model_unlearn = copy.deepcopy(model_og)
    model_retrained = ImageTextModel.from_pretrained(config.retrained_model_path).to(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrained_dir)

    # freeze original and retrained
    for param in model_og.parameters():
        param.requires_grad = False

    for param in model_retrained.parameters():
        param.requires_grad = False

    # enable gradient updates for unlearning
    for param in model_unlearn.parameters():
        param.requires_grad = True

    # ------------------------------------------- OPTIMIZATION SETUP ------------------------------------------- 
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model_unlearn.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", threshold=1e-6)

    # ------------------------------------------- BUILD DATASET ------------------------------------------- 
    dataset, num_labels = build_dataset(config, tokenizer, image_noise_params=image_noise_params)

    # ------------------------------------------- START UNLEARNING ------------------------------------------- 
    retain_dataloader, forget_dataloader, val_dataloader, test_dataloader = unlearn(
        config, output_dir, device, model_og, model_unlearn, model_retrained, optimizer, 
        optimizer_grouped_parameters, scheduler, tokenizer, dataset, num_labels, alpha, beta, theta, gamma
    )

    # Evaluate the model - the final weights only, perform manually.
#     metrics = evaluate(
#     model_og, model_unlearn, model_retrained, 
#     retain_dataloader, forget_dataloader, val_dataloader, test_dataloader, 
#     device, args, mode='test')
    # flattened_metrics = flatten_metrics(metrics)
    # wandb.log(flattened_metrics)

if __name__ == '__main__':
    main()
    wandb.finish()
    