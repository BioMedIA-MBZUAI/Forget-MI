import os
import cv2
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
import random

import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn import Softmax,LogSoftmax

from scripts import metrics as eval_metrics
from scripts import main_utils, parser


from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertForMaskedLM, BertConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from joint_img_txt.model import loss as custom_loss
from joint_img_txt.model import model_utils
from joint_img_txt.model.model_utils import CXRImageTextDataset, EdemaClassificationProcessor, RandomTranslateCrop, CenterCrop
from joint_img_txt.model.model import ImageTextModel
from joint_img_txt.model.convert_examples_to_features import convert_examples_to_features, convert_examples_to_features_multilabel

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def build_dataset(args, tokenizer, noise_params):
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
        logger.info("Loading features from cached file %s", cached_features_file)
        print("Loading features from cached file %s"%cached_features_file)
        features = torch.load(cached_features_file)
        noisy_features = torch.load(cached_noisy_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.text_data_dir)
        label_list = processor.get_labels()
        examples = processor.get_all_examples(args.text_data_dir)
        noisy_examples = processor.get_noisy_examples(args.text_data_dir)

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


    retain_img_labels, retain_img_txt_ids, test_img_labels, test_img_txt_ids, rand_img_labels, rand_img_txt_ids, \
        forget_img_labels, forget_img_txt_ids = data_split(args.data_split_path, args.forget_set_path, args.random_point_ratio)

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

    test_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, test_img_txt_ids, args.img_data_dir, 
                                  test_img_labels, args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    rand_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  noisy_txt_tokens, noisy_txt_masks, noisy_txt_segments, 
                                  noisy_txt_labels, rand_img_txt_ids, args.img_data_dir, 
                                  rand_img_labels, args.data_split_path, transform=xray_transform, perturb_img=True,
                                  noise_params=noise_params, output_channel_encoding = args.output_channel_encoding)
                                  
    print("Length of the random dataset is ", len(rand_dataset))
    print("Length of the forget dataset is ", len(test_dataset))

    dataset = {
        'rand': rand_dataset,
        'test': test_dataset
    }

    return dataset, num_labels
    
    
def data_split(split_list_path, forget_ids_path, rand_ratio):
    """Extracting finding labels
    """

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
        row = next(train_label_file_reader)
        '''
        row = 
            0 - subject id
            1 - txt id
            2 - img id
            3 - edeme severity
            4 - fold
        '''
        for row in train_label_file_reader:
            if row[0] in forget_ids:
                forget_labels[row[2]] = [float(row[3])]
                forget_img_txt_ids[row[2]] = row[1]
            else:
                if row[-1] == 'TEST':
                    test_labels[row[2]] = [float(row[3])]
                    test_img_txt_ids[row[2]] = row[1]
                else:
                    train_labels[row[2]] = [float(row[3])]
                    train_img_txt_ids[row[2]] = row[1]


    # Random sampling of datapoints for noising from the training data
    train_keys = np.array(list(train_labels.keys()))
    n_train_keys = train_keys.shape[0]
    n_rand = int(n_train_keys * rand_ratio)
    rand_idx = random.sample(range(n_train_keys), n_rand)
    rand_keys = train_keys[rand_idx]
    rand_report_ids = []
    for key in rand_keys:
        rand_labels[key] = train_labels[key]
        rand_img_txt_ids[key] = train_img_txt_ids[key]
        rand_report_ids.append(train_img_txt_ids[key])
    
    print("Total number of training labels: ", len(train_labels))
    print("Total number of training DICOM IDs: ", len(train_img_txt_ids))
    print("Total number of testing labels: ", len(rand_labels))
    print("Total number of testing DICOM IDs: ", len(rand_img_txt_ids))
    print("Total number of unlearning labels: ", len(forget_labels))
    print("Total number of unlearning DICOM IDs: ", len(forget_img_txt_ids))
    print("Total number of random labels: ", len(rand_labels))
    print("Total number of random DICOM IDs: ", len(rand_img_txt_ids))

    return train_labels, train_img_txt_ids, test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids, forget_labels, forget_img_txt_ids

def get_model_inputs(args, dataset, device, add_noise=False, noise_params=None):

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

def get_multimodal_losses(args, img_embedding, txt_embedding, img_logits, txt_logits, label_raw, report_id, num_labels, device):
    '''
    Adjust the cross entropy loss function for different label encoding options
    '''
    if args.output_channel_encoding == 'multilabel' and \
            args.training_mode != 'semisupervised_phase1':
        label_ordinal = label_onehot_or_ordinal
        # Replace the image label with the ordinally encoded label

        BCE_loss_criterion = BCEWithLogitsLoss()
        img_loss = BCE_loss_criterion(img_logits.view(-1, num_labels), 
                                        label_ordinal.view(-1, num_labels).float())
        txt_loss = BCE_loss_criterion(txt_logits.view(-1, num_labels), 
                                        label_ordinal.view(-1, num_labels).float())
    elif args.output_channel_encoding == 'multiclass' and \
            args.training_mode != 'semisupervised_phase1':
        label = label_raw

        CrossEntropyCriterion = CrossEntropyLoss() 
        # In this case, softmax is added in the model 
        # and the CrossEntropyCriterion only accepts raw labels 0-3
        img_loss = CrossEntropyCriterion(img_logits.view(-1, num_labels),
                                            label.view(-1).long())
        txt_loss = CrossEntropyCriterion(txt_logits.view(-1, num_labels),
                                            label.view(-1).long())

    '''
    Define loss functions
    '''
    if args.joint_loss_method == 'l2':
        joint_loss_criterion = torch.nn.MSELoss()
        joint_loss = joint_loss_criterion(img_embedding, txt_embedding)
    elif args.joint_loss_method == 'cosine':
        joint_loss_criterion = torch.nn.CosineEmbeddingLoss()
        y = torch.ones(img_embedding.shape[0], device=device) 
        y.requires_grad = False
        joint_loss = joint_loss_criterion(x1=img_embedding, x2=txt_embedding, y=y) 
        # y is ones so the joint loss is the negative inverse of cosine
    elif args.joint_loss_method == 'dot':
        joint_loss = custom_loss.dot_product_loss(img_embedding,
                                                    txt_embedding)
    elif args.joint_loss_method == 'ranking':
        joint_loss = custom_loss.ranking_loss(img_embedding, txt_embedding, label_raw, report_id, similarity_function=args.joint_loss_similarity_function)

    return img_loss, txt_loss, joint_loss

def get_curr_noise_loss(args, device, model, tokenizer, dataset, noise_params, num_labels):

    test_set, rand_set = dataset['test'], dataset['rand']

    print('Retrieving Test set data of length ', len(test_set))
    test_sampler = RandomSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, 
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers, 
                                  pin_memory=True)


    print('Retrieving Random set data of length ', len(rand_set))
    rand_sampler = RandomSampler(rand_set)
    rand_dataloader = DataLoader(rand_set, sampler=rand_sampler, 
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers, 
                                  pin_memory=True)

    loss_criterion = torch.nn.L1Loss()

    n_batches = 0
    rand_img_loss, rand_txt_loss, rand_joint_loss = 0.0, 0.0, 0.0
    rand_iterator = tqdm(rand_dataloader, desc="Random Set Iteration")
    for step, batch in enumerate(rand_iterator):
        n_batches += 1
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        rand_inputs, rand_labels, rand_report_id = get_model_inputs(args, batch, device, add_noise=True, noise_params=noise_params)

        rand_outputs = model(**rand_inputs)
        rand_img_emb, rand_img_log, rand_txt_emb, rand_txt_log = rand_outputs[:4]
        batch_rand_img_loss, batch_rand_txt_loss, batch_rand_joint_loss = get_multimodal_losses(args, rand_img_emb, \
                            rand_txt_emb, rand_img_log, rand_txt_log, rand_labels, rand_report_id, num_labels, device)
        
        rand_img_loss += batch_rand_img_loss
    
    rand_img_loss /= n_batches

    n_batches = 0
    test_img_loss, test_txt_loss, test_joint_loss = 0.0, 0.0, 0.0
    test_iterator = tqdm(test_dataloader, desc="Test Set Iteration")
    for step, batch in enumerate(test_iterator):
        n_batches += 1
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        test_inputs, test_labels, test_report_id = get_model_inputs(args, batch, device)
        test_outputs = model(**test_inputs)
        test_img_emb, test_img_log, test_txt_emb, test_txt_log = test_outputs[:4]
        batch_test_img_loss, batch_test_txt_loss, batch_test_joint_loss = get_multimodal_losses(args, test_img_emb, \
                            test_txt_emb, test_img_log, test_txt_log, test_labels, test_report_id, num_labels, device)

        test_img_loss += batch_test_img_loss
    
    test_img_loss /= n_batches

    noise_loss = loss_criterion(rand_img_loss, test_img_loss)

    return rand_img_loss, test_img_loss, noise_loss

    # Perform unlearning on forget set using preprocessed encodings for computing losses.


def main():
    
    random.seed(0)
    args = parser.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)

    assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

    training_epoch_to_load = 300
    # load original joinxray model from checkpoint
    
    model = ImageTextModel.from_pretrained(f'output/cross_val/model/supervised/example/checkpoints/checkpoint-{training_epoch_to_load}/')
    model = model.to(device)

    print('Model loaded successfully')

    print('Starting the Noising Process...')

    with torch.no_grad():
        for int_mean in range(11):
            mean = int_mean / 10
            for int_std in range(11):
                std = int_std / 10

                noise_params = {
                    'mean': mean,
                    'std': std

                }

                dataset, num_labels = build_dataset(args, tokenizer, noise_params)

                rand_loss, test_loss, noise_loss = get_curr_noise_loss(args, device, model, tokenizer, dataset, noise_params, num_labels)

                print(f'For mean={mean} and std={std}, \n \t Rand loss = {rand_loss} \n \t Test_loss = {test_loss} \n \t Noise_loss={noise_loss}')


if __name__ == '__main__':
    main()
    