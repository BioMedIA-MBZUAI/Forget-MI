from datetime import datetime
import os
import random
import numpy as np
import torch
from joint_img_txt.model.model import ImageTextModel
from joint_img_txt.model.convert_examples_to_features import convert_examples_to_features, convert_examples_to_features_multilabel
from joint_img_txt.model.model_utils import CXRImageTextDataset, EdemaClassificationProcessor, RandomTranslateCrop, CenterCrop
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertForMaskedLM, BertConfig, AutoModel, AutoTokenizer
from torch.nn.functional import softmax
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, Sampler
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import logging
from scipy.stats import logistic
from scripts.metrics import compute_auc, get_acc_f1, compute_mse
from scripts.evaluate_unlearning import get_probability_measure
import torch.nn.functional as F

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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    retain_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, retain_img_txt_ids, args.img_data_dir, 
                                  retain_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform,
                                  output_channel_encoding = args.output_channel_encoding)

    test_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, test_img_txt_ids, args.img_data_dir, 
                                  test_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    rand_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  noisy_txt_tokens, noisy_txt_masks, noisy_txt_segments, 
                                  noisy_txt_labels, rand_img_txt_ids, args.img_data_dir, 
                                  rand_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, perturb_img=True,
                                  noise_params=image_noise_params, output_channel_encoding = args.output_channel_encoding)

    forget_dataset = CXRImageTextDataset(args.img_localdisk_data_dir, args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, forget_img_txt_ids, args.img_data_dir, 
                                  forget_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    val_dataset = CXRImageTextDataset( args.img_localdisk_data_dir, args.id, 
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

@torch.no_grad()
def collect_logits(dataset, model, device, batch_size=32):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits = []
    for batch in data_loader:
        inputs, _, _ = get_model_inputs(args, batch, device)
        outputs = model(**inputs)
        logits = outputs[1].detach().cpu()
        all_logits.append(logits)

    print(f"Collected logits for {len(all_logits)}")
    return torch.cat(all_logits, axis=0)

def calculate_entropy_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  
    return entropy

def prepare_mia_data(retain_set, test_set, forget_set, model, device, batch_size=32):
    print("Collecting logits and calculating entropy...")
    
    # Collect logits
    retain_logits = collect_logits(retain_set, model, device, batch_size)
    test_logits = collect_logits(test_set, model, device, batch_size)
    forget_logits = collect_logits(forget_set, model, device, batch_size)

    # Calculate probabilities from logits
    retain_probs = softmax(torch.tensor(retain_logits), dim=1).numpy()
    test_probs = softmax(torch.tensor(test_logits), dim=1).numpy()
    forget_probs = softmax(torch.tensor(forget_logits), dim=1).numpy()

    # Calculate entropy
    retain_entropy = calculate_entropy_from_logits(torch.tensor(retain_probs)).numpy()
    test_entropy = calculate_entropy_from_logits(torch.tensor(test_probs)).numpy()
    forget_entropy = calculate_entropy_from_logits(torch.tensor(forget_probs)).numpy()

    # Prepare data for SVC
    X_train_logits = np.concatenate([retain_logits, test_logits])
    X_train_entropy = np.concatenate([retain_entropy, test_entropy]).reshape(-1, 1)
    Y_train = np.concatenate([np.ones(len(retain_logits)), np.zeros(len(test_logits))])
    X_forget_logits = forget_logits
    X_forget_entropy = forget_entropy.reshape(-1, 1)

    return X_train_logits, X_train_entropy, Y_train, X_forget_logits, X_forget_entropy

def train_mia_classifier(X_train, Y_train, X_forget):
    clf = SVC(C=3, kernel="rbf", gamma="auto")
    print("Training SVC classifier...")
    clf.fit(X_train, Y_train)

    forget_predictions = clf.predict(X_forget)
    membership_score = forget_predictions.mean()
    return membership_score, forget_predictions

def run_mia(retain_set, test_set, forget_set, model, device, batch_size=32):
    print("Preparing data for MIA...")
    X_train_logits, X_train_entropy, Y_train, X_forget_logits, X_forget_entropy = prepare_mia_data(
        retain_set, test_set, forget_set, model, device, batch_size
    )

    print("Running MIA via logits...")
    mia_score_logits, forget_predictions_logits = train_mia_classifier(X_train_logits, Y_train, X_forget_logits)

    print("Running MIA via entropy...")
    mia_score_entropy, forget_predictions_entropy = train_mia_classifier(X_train_entropy, Y_train, X_forget_entropy)

    return mia_score_logits, mia_score_entropy


def update_mia_results_csv(csv_path, base_model_path, mia_score):
    # ----- CREATE CSV IF IT DOESN'T EXIST -----
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["BaseModelPath", "MIA_Score"])

    # ----- COLUMNs are the model name and MIA score -----
    new_row = {"BaseModelPath": base_model_path, "MIA_Score": mia_score}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Updated MIA results saved to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Membership Inference Attack")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--bert_pretrained_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--text_data_dir', type=str, required=True)
    parser.add_argument('--synonyms_dir', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, default=320)
    parser.add_argument('--output_channel_encoding', type=str, default='multiclass')
    parser.add_argument('--reprocess_input_data', action='store_true')
    parser.add_argument('--img_localdisk_data_dir', type=str, required=True)
    parser.add_argument('--id', type=str, default='dataset_id')
    parser.add_argument('--img_data_dir', type=str, required=True)
    parser.add_argument('--data_split_path', type=str, required=True)
    parser.add_argument('--forget_set_path', type=str, required=True)
    parser.add_argument('--random_point_ratio', type=float, default=0.1)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--bert_pool_last_hidden', action='store_true')
    parser.add_argument('--bert_pool_use_img', action='store_true')
    parser.add_argument('--bert_pool_img_lowerlevel', action='store_true')
    return parser.parse_args()

args = parse_args()

# ---------- SET RANDOM SEED -------
set_seed(args.random_seed)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------- LOAD UNLEARNED MODEL AND TOKENIZER -------
model_og = ImageTextModel.from_pretrained(args.base_model_path).to(device)
model_og.eval()
tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)

# ---------- BUILD DATASET AND CREATE LOADER-------
dataset, num_labels = build_dataset(args, tokenizer)
retain_set, val_set, rand_set, forget_set, test_set = (
    dataset['retain'], dataset['validation'], dataset['random'], dataset['forget'], dataset['test']
)    
# ---------- RUN MIA -----------
mia_score_logits, mia_score_entropy = run_mia(
    retain_set, test_set, forget_set, model_og, device, args.batch_size
)
# ---------- UPDATE MIA RESULTS -------
csv_path = os.path.join(args.output_dir, "mia_results.csv")
update_mia_results_csv(csv_path, args.base_model_path + "_logits", mia_score_logits)
update_mia_results_csv(csv_path, args.base_model_path + "_entropy", mia_score_entropy)

print(f"MIA scores saved: logits = {mia_score_logits}, entropy = {mia_score_entropy}")