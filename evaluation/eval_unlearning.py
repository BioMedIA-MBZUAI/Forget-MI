import os
import argparse
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.svm import SVC
from transformers import BertTokenizer, BertConfig

from joint_img_txt.model.model import ImageTextModel
from joint_img_txt.model import loss as custom_loss
from training.forgetme_partial import get_model_inputs, set_seed, build_dataset
from evaluation.eval_utils import get_probability_measure, compute_metrics


class CombinedLoss(nn.Module):
    def __init__(self, num_labels):
        super(CombinedLoss, self).__init__()
        self.num_labels = num_labels
        self.ce_loss = nn.CrossEntropyLoss()
        self.ranking_loss = custom_loss.ranking_loss

    def forward(
        self, img_logits, txt_logits, img_embedding, txt_embedding, labels, report_id
    ):
        img_loss = self.ce_loss(
            img_logits.view(-1, self.num_labels), labels.view(-1).long()
        )
        txt_loss = self.ce_loss(
            txt_logits.view(-1, self.num_labels), labels.view(-1).long()
        )
        joint_loss = self.ranking_loss(
            img_embedding, txt_embedding, labels, report_id, "dot"
        )
        total_loss = img_loss + txt_loss + joint_loss
        return total_loss, img_loss, txt_loss, joint_loss


@torch.no_grad()
def collect_logits(dataset, model, device, batch_size=32):
    criterion_cls = CombinedLoss(4)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    img_logits = []
    sample_losses = []
    for batch in data_loader:
        inputs, labels, report_id = get_model_inputs(args, batch, device)
        labels = labels.to(device)

        outputs = model(**inputs)
        image_logits = outputs[1].to(device)
        img_logits.append(image_logits.detach().cpu())
        image_embeddings = outputs[0].to(device)
        text_logits = outputs[3].to(device)
        text_embeddings = outputs[2].to(device)

        total_loss_cls, img_loss, txt_loss, joint_loss = criterion_cls(
            image_logits,
            text_logits,
            image_embeddings,
            text_embeddings,
            labels,
            report_id,
        )
        sample_losses.append(img_loss.view(1).detach().cpu())

    img_logits = torch.cat(img_logits, axis=0)
    sample_losses = torch.cat(sample_losses, axis=0)

    return img_logits, sample_losses


def prepare_mia_data(
    retain_set, test_set, forget_set, model, device, batch_size=32, model_name="model"
):
    print("Collecting logits and calculating entropy...")

    retain_losses = collect_logits(retain_set, model, device, batch_size)
    test_losses = collect_logits(test_set, model, device, batch_size)
    forget_losses = collect_logits(forget_set, model, device, batch_size)

    X_train_losses = np.concatenate(
        [retain_losses.numpy(), test_losses.numpy()]
    ).reshape(-1, 1)
    X_forget_losses = forget_losses.numpy().reshape(-1, 1)
    Y_train_losses = np.concatenate(
        [np.ones(len(retain_losses)), np.zeros(len(test_losses))]
    )
    print(f"X_train_losses shape: {X_train_losses.shape}")
    print(f"X_forget_losses shape: {X_forget_losses.shape}")
    print(f"Y_train_losses shape: {Y_train_losses.shape}")

    np.save(
        f"{model_name}_losses.npy",
        {
            "retain": retain_losses.numpy(),
            "test": test_losses.numpy(),
            "forget": forget_losses.numpy(),
        },
    )
    print(f"Losses saved to {model_name}_losses.npy")
    return (
        X_train_losses,
        X_forget_losses,
        Y_train_losses,
    )


def train_mia_classifier(X_train, Y_train, X_forget):
    print("Training SVC classifier...")
    clf = SVC(C=3, kernel="rbf", gamma="auto")
    clf.fit(X_train, Y_train)

    forget_predictions = clf.predict(X_forget)
    membership_score = forget_predictions.mean()

    return membership_score, forget_predictions


def run_mia(
    retain_set,
    test_set,
    forget_set,
    model,
    device,
    batch_size=32,
    model_name="model",  # model name e.g. Forget-MI-10
):
    print("Preparing data for MIA...")
    (
        X_train_losses,
        X_forget_losses,
        Y_train_losses,
    ) = prepare_mia_data(
        retain_set, test_set, forget_set, model, device, batch_size, model_name
    )

    mia_score_losses, _ = train_mia_classifier(
        X_train_losses, Y_train_losses, X_forget_losses
    )

    return mia_score_losses


def parse_args():
    parser = argparse.ArgumentParser(description="Membership Inference Attack")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--bert_pretrained_dir", type=str, required=True)
    parser.add_argument("--retrained_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_split_path", type=str, required=True)
    parser.add_argument("--forget_set_path", type=str, required=True)
    parser.add_argument("--synonyms_dir", type=str, required=True)

    # For dataset building
    parser.add_argument("--bert_pool_last_hidden", action="store_true")
    parser.add_argument("--bert_pool_use_img", action="store_true")
    parser.add_argument("--bert_pool_img_lowerlevel", action="store_true")
    parser.add_argument("--output_channel_encoding", type=str, default="multiclass")
    parser.add_argument("--max_seq_length", type=int, default=320)
    parser.add_argument("--reprocess_input_data", action="store_true")
    parser.add_argument("--random_point_ratio", type=float, default=0.1)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--id", type=str, default="dataset_id")
    parser.add_argument("--img_data_dir", type=str, required=True)

    return parser.parse_args()


args = parse_args()

# ---------- SET RANDOM SEED -------
set_seed(args.random_seed)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------- LOAD UNLEARNED MODEL AND TOKENIZER -------
config_path = f"{args.config}"
with open(config_path, "r") as config_file:
    config = json.load(config_file)
text_model_config = BertConfig.from_dict(config)

model_ul = ImageTextModel(text_model_config).to(device)

model_ul.load_state_dict(torch.load(args.base_model_path, map_location=device))
model_ul.to(device)

model_ul.eval()

tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)

# ---------- BUILD DATASET AND CREATE LOADER-------
dataset, num_labels = build_dataset(args, tokenizer)
retain_set, val_set, rand_set, forget_set, test_set = (
    dataset["retain"],
    dataset["validation"],
    dataset["random"],
    dataset["forget"],
    dataset["test"],
)
# ---------- RUN MIA -----------
mia_score_losses = run_mia(
    retain_set,
    test_set,
    forget_set,
    model_ul,
    device,
    args.batch_size,
    args.base_model_path,
)
print(f"MIA Score (Losses): {mia_score_losses}")

# ---------- SIMILIARITY MEASURE -------
model_re = ImageTextModel.from_pretrained(args.retrained_model_path).to(device)
model_re.eval()
retain_data_loader = DataLoader(retain_set, batch_size=args.batch_size, shuffle=False)
similarity_measure = get_probability_measure(
    args, model_ul, model_re, retain_data_loader, device
)
print(f"Similarity measure between original and retrained model: {similarity_measure}")

# ---------- PERFORMANCE -------
print("Test set:")
test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
test_metrics = compute_metrics(model_ul, test_data_loader, device, args)

print("Forget set:")
forget_data_loader = DataLoader(forget_set, batch_size=args.batch_size, shuffle=False)
forget_metrics = compute_metrics(model_ul, forget_data_loader, device, args)

# ---------- SAVE -------
output_data = {
    "BaseModelPaths": args.base_model_path,
    "1-SimilarityMeasure": round(1 - similarity_measure, 3),
    "MIA_Losses": round(mia_score_losses, 3),
    "Forget_AUC": (
        round(np.mean(forget_metrics["AUC"]), 3) if "AUC" in forget_metrics else None
    ),
    "Forget_Macro_F1": (
        round(forget_metrics["Macro F1"], 3) if "Macro F1" in forget_metrics else None
    ),
    "Test_AUC": (
        round(np.mean(test_metrics["AUC"]), 3) if "AUC" in test_metrics else None
    ),
    "Test_Macro_F1": (
        round(test_metrics["Macro F1"], 3) if "Macro F1" in test_metrics else None
    ),
    "Forget Pairwise AUC": forget_metrics["Pairwise AUC"],
    "Test Pairwise AUC": test_metrics["Pairwise AUC"],
    "Forget F1": forget_metrics["F1"],
    "Test F1": test_metrics["F1"],
}

output_file = os.path.join(args.output_dir, f"mia_score_{args.id}.csv")
os.makedirs(args.output_dir, exist_ok=True)
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(output_data.keys())
    writer.writerow(output_data.values())
print(f"Results saved to {output_file}")
