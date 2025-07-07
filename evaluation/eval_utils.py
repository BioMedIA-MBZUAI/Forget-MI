"""
Utils for evaluating unlearning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from scipy.stats import logistic
from scripts.metrics import compute_auc, compute_mse, get_acc_f1

from training.forgetme_partial import get_model_inputs


def get_probability_measure(args, model_ul, model_og, retain_dataloader, device="cuda"):
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


def compute_metrics(model, data_loader, device="cuda", args=None):
    """Compute performance metrics for a given model and data loader."""
    labels_all = []
    logits_all = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            inputs, labels, _ = get_model_inputs(args, batch, device)

            outputs = model(**inputs)
            logits = outputs[1]

            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    if logits_all.ndim == 1:
        logits_all = logits_all.reshape(-1, args.num_classes)

    if args.output_channel_encoding == "multilabel":
        preds_probs = logistic.cdf(logits_all)
    else:
        preds_probs = softmax(logits_all, axis=1)

    preds_all = preds_probs
    # if np.any(np.isnan(preds_probs)):
    #     print(preds_all)
    #     print("FIX: NaN values found in predictions")

    auc, pairwise_auc = compute_auc(
        labels_all, preds_probs, output_channel_encoding=args.output_channel_encoding
    )
    acc_f1_metrics, _, _ = get_acc_f1(
        labels_all, preds_probs, args.output_channel_encoding
    )
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
        "labels": labels_all,
    }
