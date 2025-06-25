"""
Module for evaluating unlearning.
"""

from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import numpy as np
import torch
import random
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joint_img_txt.model.model import ImageTextModel
from scripts.metrics import compute_auc, get_acc_f1, compute_mse
from tqdm import tqdm
from scipy.stats import logistic
from scipy.special import softmax
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_probability_measure(args, model_ul, model_og, retain_dataloader, device='cuda'):
    """
    Compute cosine similarity between logits of model_ul and model_og on the retain set.
    """
    model_ul.eval()
    model_og.eval()

    similarities = []

    with torch.no_grad():
        for batch in retain_dataloader:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

            inputs, _, _ = get_model_inputs(args, batch, device) 

            outputs_ul = model_ul(**inputs)
            outputs_og = model_og(**inputs)

            logits_ul = outputs_ul[1] 
            logits_og = outputs_og[1]  

            similarity = F.cosine_similarity(logits_ul, logits_og, dim=1) 
            similarities.append(similarity.cpu().numpy())

    similarities = np.concatenate(similarities)

    mean_similarity = np.mean(similarities)
    return mean_similarity

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

def compute_js_divergence(p, q):
    """Compute Jensen-Shannon Divergence between two probability distributions"""
    return jensenshannon(p, q)

def compute_kl_divergence(p, q):
    """Compute KL Divergence between two probability distributions"""
    return entropy(p, q)

def collect_entropy(args, data_loader, model, device='cuda'):
    """Collect entropy of model predictions"""
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            inputs, labels, _ = get_model_inputs(args, batch, device)

            outputs = model(**inputs)
            logits = outputs[1]  # Logits from the model
            probs = softmax(logits, axis=1)
            prob.append(entropy(probs, axis=1))  # Calculate entropy over probabilities

    return np.concatenate(prob)
    
def compute_metrics(model, data_loader, device='cuda', args=None):
    """Compute performance metrics for a given model and data loader."""
    labels_all = []
    logits_all = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            inputs, labels, _ = get_model_inputs(args, batch, device)

            outputs = model(**inputs)
            logits = outputs[1]  # Logits from the model

            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    # Ensure logits are reshaped correctly if 1D
    if logits_all.ndim == 1:
        logits_all = logits_all.reshape(-1, args.num_classes)

    # Compute probabilities
    if args.output_channel_encoding == 'multilabel':
        preds_probs = logistic.cdf(logits_all)  # Apply logistic for multilabel classification
    else:
        preds_probs = softmax(logits_all, axis=1)  # Apply softmax for multiclass classification

    preds_all = preds_probs
    if np.any(np.isnan(preds_probs)):
        print(preds_all)
        print("FIX: NaN values found in predictions")

    # Compute metrics
    auc, pairwise_auc = compute_auc(labels_all, preds_probs, output_channel_encoding=args.output_channel_encoding)
    acc_f1_metrics, _, _ = get_acc_f1(labels_all, preds_probs, args.output_channel_encoding)
    mse_val = compute_mse(logits_all, labels_all, args.output_channel_encoding)

    return {
        "AUC": auc,
        "Pairwise AUC": pairwise_auc,
        "Accuracy": acc_f1_metrics["accuracy"],
        "F1": acc_f1_metrics["f1"],
        "Precision": acc_f1_metrics["precision"],
        "Recall": acc_f1_metrics["recall"],
        "Macro F1": acc_f1_metrics["macro_f1"],
        "MSE": mse_val,
        "logits": logits_all,
        "preds_all": preds_all,
        "labels": labels_all
    }

def filter_metrics(metrics):
    """Remove unnecessary keys (logits, preds_all, labels) from the metrics dictionary."""
    keys_to_remove = ["logits", "preds_all", "labels"]
    return {key: value for key, value in metrics.items() if key not in keys_to_remove}

def evaluate(model_og, model_ul, model_re, retain_loader, forget_loader, val_dataloader, test_dataloader, device, args, cached_metrics = None, mode = 'val'):
    """Evaluate unlearning using the original and unlearned models on the retain, forget, and test sets"""
    set_seed(args.random_seed)

    model_og.eval()
    model_ul.eval()
    model_re.eval()

    results = {}

    if mode == 'val':

        if cached_metrics is None:
            random_model = ImageTextModel.from_pretrained(args.random_model_path)
            random_model.to(device)
            random_model.eval()

            print("Validation Performance Evaluation")
            val_metrics_og = compute_metrics(model_og, val_dataloader, device, args)
            val_metrics_re = compute_metrics(model_re, val_dataloader, device, args)
            val_metrics_random = compute_metrics(random_model, val_dataloader, device, args)

            print("Forget Performance Evaluation")
            forget_metrics_og = compute_metrics(model_og, forget_loader, device, args)
            forget_metrics_re = compute_metrics(model_re, forget_loader, device, args)
            forget_metrics_random = compute_metrics(random_model, forget_loader, device, args)

            print("Test Performance Evaluation")
            test_metrics_og = compute_metrics(model_og, test_dataloader, device, args)
            test_metrics_re = compute_metrics(model_re, test_dataloader, device, args)
            test_metrics_random = compute_metrics(random_model, test_dataloader, device, args)

            # Cache metrics for reuse in the future epochs
            cached_metrics = {
                'val_metrics_og': val_metrics_og,
                'val_metrics_re': val_metrics_re,
                'val_metrics_random': val_metrics_random,
                'forget_metrics_og': forget_metrics_og,
                'forget_metrics_re': forget_metrics_re,
                'forget_metrics_random': forget_metrics_random,
                'test_metrics_og': test_metrics_og,
                'test_metrics_re': test_metrics_re,
                'test_metrics_random': test_metrics_random
            }
        # If metrics are cached, just retrieve them
        else:
            val_metrics_og = cached_metrics['val_metrics_og']
            val_metrics_re = cached_metrics['val_metrics_re']
            val_metrics_random = cached_metrics['val_metrics_random']
            forget_metrics_og = cached_metrics['forget_metrics_og']
            forget_metrics_re = cached_metrics['forget_metrics_re']
            forget_metrics_random = cached_metrics['forget_metrics_random']
            test_metrics_og = cached_metrics['test_metrics_og']
            test_metrics_re = cached_metrics['test_metrics_re']
            test_metrics_random = cached_metrics['test_metrics_random']

        val_metrics_ul = compute_metrics(model_ul, val_dataloader, device, args)

        forget_metrics_ul = compute_metrics(model_ul, forget_loader, device, args)

        test_metrics_ul = compute_metrics(model_ul, test_dataloader, device, args)

        js_forget = compute_js_divergence(forget_metrics_og["preds_all"], forget_metrics_ul["preds_all"])
        kl_forget = compute_kl_divergence(forget_metrics_og["preds_all"], forget_metrics_ul["preds_all"])
        
        preds_forget_og = forget_metrics_og["preds_all"]
        preds_forget_ul = forget_metrics_ul["preds_all"]
        activation_distance = np.mean(np.abs(forget_metrics_og["logits"] - forget_metrics_ul["logits"]))

        entropy_forget_og = np.mean(entropy(preds_forget_og, axis=1))
        entropy_forget_ul = np.mean(entropy(preds_forget_ul, axis=1))

        # look at definition at https://arxiv.org/pdf/2205.08096
        forget_zrf_score_ul_ran = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(test_metrics_ul["preds_all"], test_metrics_random["preds_all"])])    
        forget_zrf_score_ul_re = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(test_metrics_ul["preds_all"], test_metrics_re["preds_all"])])    
        test_zrf_score_ul_re = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(test_metrics_ul["preds_all"], test_metrics_re["preds_all"])])    
        test_zrf_score_ul_ran = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(test_metrics_ul["preds_all"], test_metrics_random["preds_all"])])    

        results.update({
            "val_metrics_og": filter_metrics(val_metrics_og),
            "val_metrics_ul": filter_metrics(val_metrics_ul),
            "val_metrics_re": filter_metrics(val_metrics_re),
            "val_metrics_random": filter_metrics(val_metrics_random),
            "forget_metrics_og": filter_metrics(forget_metrics_og),
            "forget_metrics_ul": filter_metrics(forget_metrics_ul),
            "forget_metrics_re": filter_metrics(forget_metrics_re),
            "forget_metrics_random": filter_metrics(forget_metrics_random),
            "test_metrics_og": filter_metrics(test_metrics_og),
            "test_metrics_ul": filter_metrics(test_metrics_ul),
            "test_metrics_re": filter_metrics(test_metrics_re),
            "test_metrics_random": filter_metrics(test_metrics_random),
            "test_metrics_random": test_metrics_random,
            "js_forget": np.mean(js_forget),
            "kl_forget": np.mean(kl_forget),
            "activation_distance": activation_distance,
            "entropy_forget_og": entropy_forget_og,
            "entropy_forget_ul": entropy_forget_ul,
            "forget_zrf_score_ul_ran": forget_zrf_score_ul_ran,
            "forget_zrf_score_ul_re": forget_zrf_score_ul_re,
            "test_zrf_score_ul_re": test_zrf_score_ul_re,
            "test_zrf_score_ul_ran": test_zrf_score_ul_ran
        })

        return results, cached_metrics
    else:
        # TODO
        return None
        # print("-"*20, "PERFORMANCE METRICS", "-"*20)
        # retain_metrics_og = compute_metrics(model_og, val_dataloader, device, args)
        # retain_metrics_ul = compute_metrics(model_ul, val_dataloader, device, args)

        # forget_metrics_og = compute_metrics(model_og, forget_loader, device, args)
        # forget_metrics_ul = compute_metrics(model_ul, forget_loader, device, args)
        # forget_metrics_re = compute_metrics(model_re, forget_loader, device, args)
        # forget_metrics_random = compute_metrics(random_model, forget_loader, device, args)

        # test_metrics_og = compute_metrics(model_og, val_dataloader, device, args)
        # test_metrics_ul = compute_metrics(model_ul, val_dataloader, device, args)

        # print("-"*20, "UNLEARNING METRICS", "-"*20)
        # # Lower JS Divergence indicates more similarity in predictions, higher values show divergence
        # js_forget = compute_js_divergence(forget_metrics_og["preds_all"], forget_metrics_ul["preds_all"])
        # js_test = compute_js_divergence(test_metrics_og["preds_all"], test_metrics_ul["preds_all"])
        
        # #------------------- KL Divergence --------------------
        # kl_forget = compute_kl_divergence(forget_metrics_og["preds_all"], forget_metrics_ul["preds_all"])
        # kl_test = compute_kl_divergence(test_metrics_og["preds_all"], test_metrics_ul["preds_all"])
        
        # activation_distance_og_ul = np.mean(np.abs(forget_metrics_og["logits"] - forget_metrics_ul["logits"]))
        # activation_distance_re_ul = np.mean(np.abs(forget_metrics_re["logits"] - forget_metrics_ul["logits"]))
        
        # entropy_forget_og = np.mean(entropy(forget_metrics_og["preds_all"], axis=1))
        # entropy_forget_ul = np.mean(entropy(forget_metrics_ul["preds_all"], axis=1))
        
        # zrf_score_ul_ran = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(forget_metrics_ul["preds_all"], forget_metrics_random["preds_all"])])    
        # zrf_score_ul_re = 1 - np.mean([compute_js_divergence(u, r) for u, r in zip(forget_metrics_ul["preds_all"], forget_metrics_re["preds_all"])])    
        
        # results.update({
        #     "retain_metrics_og": retain_metrics_og,
        #     "retain_metrics_ul": retain_metrics_ul,
        #     "forget_metrics_og": forget_metrics_og,
        #     "forget_metrics_ul": forget_metrics_ul,
        #     "forget_metrics_re": forget_metrics_re,
        #     "forget_metrics_random": forget_metrics_random,
        #     "test_metrics_og": test_metrics_og,
        #     "test_metrics_ul": test_metrics_ul,
        #     "js_forget": np.mean(js_forget),
        #     "js_test": np.mean(js_test),
        #     "kl_forget": np.mean(kl_forget),
        #     "kl_test": np.mean(kl_test),
        #     "activation_distance_og_ul": activation_distance_og_ul,
        #     "activation_distance_re_ul": activation_distance_re_ul,
        #     "entropy_forget_og": entropy_forget_og,
        #     "entropy_forget_ul": entropy_forget_ul,
        #     "zrf_score_ul_ran": zrf_score_ul_ran,
        #     "zrf_score_ul_re": zrf_score_ul_re,
        # })

        # #------------------- Membership Inference Attack ---------------
        # print("MIA")
        # models = {'og': model_og, 're': model_re, 'ul': model_ul}
        # attack_results = {}

        # for model_name, model in models.items():
        #     retain_entropy = collect_entropy(retain_loader, model, device)
        #     forget_entropy = collect_entropy(forget_loader, model, device)
        #     test_entropy = collect_entropy(val_dataloader, model, device)

        #     X_r = np.concatenate([retain_entropy, test_entropy]).reshape(-1, 1)
        #     Y_r = np.concatenate([np.ones(len(retain_entropy)), np.zeros(len(test_entropy))])

        #     clf_svm = SVC(C=3, gamma='auto', kernel='rbf')
        #     clf_logreg = LogisticRegression(class_weight='balanced', solver='lbfgs')

        #     clf_svm.fit(X_r, Y_r)
        #     clf_logreg.fit(X_r, Y_r)

        #     attack_results[model_name] = {
        #         'svm_success': np.mean(clf_svm.predict(forget_entropy.reshape(-1, 1)) == 1),
        #         'logreg_success': np.mean(clf_logreg.predict(forget_entropy.reshape(-1, 1)) == 1)
        #     }

        # results["attack_results"] = attack_results

        # return results