import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trainer import ExplanationsTrainer
from data.train_test_datasets import train_dataset, test_dataset

import evaluate

metric = evaluate.load("accuracy")


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return metric.compute(predictions=predictions, references=labels)


def get_probabilities(model, sentence_pairs):
    features = tokenizer(
        sentence_pairs, padding=True, truncation=True, return_tensors="pt"
    )
    for name in features:
        features[name] = features[name].to(device)
    with torch.no_grad():
        scores = model(
            **features
        ).logits  # 0: 'contradiction', 1: 'entailment', 0: 'neutral'
        probs = torch.nn.functional.softmax(scores, dim=-1).detach().cpu().numpy()

    return probs


def get_predictions(model, sentence_pairs):
    probs = get_probabilities(model, sentence_pairs)
    return [np.argmax(prob) for prob in probs]


def data_collator(rows):
    features = tokenizer(
        [(row["hypothesis"], row["premise"]) for row in rows],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.Tensor([row["labels"] for row in rows]).to(device)
    for name in features:
        features[name] = features[name].to(device)
    features["labels"] = labels

    return features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "answerdotai/ModernBERT-base"
classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

critic_loss_weight = 0.05
lr = 1e-5
# Define training args
training_args = TrainingArguments(
    output_dir=f"modern_bert-explanations-model-{lr}-{critic_loss_weight}",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=2,
    bf16=True,  # bfloat16 training
    optim="adamw_torch_fused",  # improved optimizer
    # logging & evaluation strategies
    logging_strategy="steps",
    gradient_checkpointing=True,
    eval_strategy="epoch",
    logging_steps=1,
    log_level="info",
    save_strategy="steps",
    save_steps=10000,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    remove_unused_columns=False,
    torch_empty_cache_steps=2,
    dataloader_pin_memory=False,
)

# Create a Trainer instance
trainer = ExplanationsTrainer(
    critic_loss_weight=critic_loss_weight,
    model=classifier,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
