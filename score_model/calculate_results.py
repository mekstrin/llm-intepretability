import os
from functools import partial
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
from data.train_test_datasets import (
    make_test_set_cose,
    make_test_set_mnli,
    make_test_set_esnli,
)
from utils import (
    get_predictions,
    zst_get_results,
    zst_get_predictions,
    get_probabilities,
)
from explanations.explanations_eval import (
    eval_explanations,
    zst_get_explanations,
    zst_predict,
)
from score_model.metrics import (
    compute_iou_for_different_top_tokens,
    comprehensiveness_nli,
    comprehensiveness_zst,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detokenizer = TreebankWordDetokenizer()

model_paths = []


def evaluate_model(model_path, model_name, datasets, num_batches=1000):
    print(f"Evaluating model: {model_name}")

    results_dir = f"results_{model_name}"
    shutil.rmtree(f"./{results_dir}", ignore_errors=True)
    os.makedirs(results_dir, exist_ok=True)

    base_model_name = "cross-encoder/nli-deberta-v3-xsmall"
    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, use_fast=False, device=device
    )
    zst_classifier = pipeline(
        "zero-shot-classification",
        tokenizer=tokenizer,
        model=model_path,
        use_fast=False,
        device=device,
    )
    classifier = classifier.to(device)

    dataset_esnli = datasets["esnli"]
    esnli_size = len(dataset_esnli["sentence_pairs"])
    predictions_esnli = []

    for i in range(num_batches):
        batch_size = esnli_size // num_batches
        prediction_batch = get_predictions(
            classifier,
            dataset_esnli["sentence_pairs"][i * batch_size : (i + 1) * batch_size],
        )
        predictions_esnli += prediction_batch

    esnli_accuracy = accuracy_score(
        dataset_esnli["test_labels"][: len(predictions_esnli)], predictions_esnli
    )
    print(f"Accuracy on ESNLI: {esnli_accuracy}")

    dataset_mnli = datasets["mnli"]
    mnli_size = len(dataset_mnli["sentence_pairs"])
    predictions_mnli = []

    for i in range(num_batches):
        batch_size = mnli_size // num_batches
        prediction_batch = get_predictions(
            classifier,
            dataset_mnli["sentence_pairs"][i * batch_size : (i + 1) * batch_size],
        )
        predictions_mnli += prediction_batch

    mnli_accuracy = accuracy_score(
        dataset_mnli["test_labels"][: len(predictions_mnli)], predictions_mnli
    )
    print(f"Accuracy on MNLI: {mnli_accuracy}")

    dataset_cose = datasets["cose"]
    cose_results = zst_get_results(
        zst_classifier,
        dataset_cose["question"],
        candidate_labels=dataset_cose["candidate_labels_list"],
    )
    predictions_cose = zst_get_predictions(
        cose_results, candidate_labels=dataset_cose["candidate_labels_list"]
    )
    cose_accuracy = accuracy_score(predictions_cose, dataset_cose["true_labels"])
    print(f"Accuracy on COS-E: {cose_accuracy}")

    explanations_esnli = eval_explanations(
        classifier, tokenizer, dataset_esnli["sentence_pairs"]
    )
    explanations_mnli = eval_explanations(
        classifier, tokenizer, dataset_mnli["sentence_pairs"]
    )
    cose_explanations = zst_get_explanations(
        zst_classifier,
        dataset_cose["question"],
        candidate_labels=dataset_cose["candidate_labels_list"],
    )

    ground_truth_list_esnli = dataset_esnli["extractive_explanations"]
    ground_truth_list_cose = dataset_cose["extractive_explanation"]
    top_tokens = list(range(1, 10))

    iou_top_k_results_esnli = compute_iou_for_different_top_tokens(
        explanations_esnli, ground_truth_list_esnli, top_tokens
    )
    iou_top_k_results_cose = compute_iou_for_different_top_tokens(
        cose_explanations, ground_truth_list_cose, top_tokens
    )

    plt.figure()
    plt.plot(top_tokens, iou_top_k_results_esnli, label="IOU")
    plt.legend(loc="upper right")
    plt.xlabel("k")
    plt.title(f"ESNLI IOU with different number of top k tokens for {model_name}")
    plt.savefig(f"{results_dir}/esnli_iou.png")
    plt.close()

    plt.figure()
    plt.plot(top_tokens, iou_top_k_results_cose, label="IOU")
    plt.legend(loc="upper right")
    plt.xlabel("k")
    plt.title(f"COSE IOU with different number of top k tokens for {model_name}")
    plt.savefig(f"{results_dir}/cose_iou.png")
    plt.close()

    bins = [0.1, 0.3, 0.5]

    mnli_comprehensiveness = []
    for sentence, explanation in zip(dataset_mnli["sentence_pairs"], explanations_mnli):
        comprehensiveness = comprehensiveness_nli(
            explanation,
            sentence,
            partial(get_probabilities, classifier, tokenizer),
            bins,
        )
        mnli_comprehensiveness.append(comprehensiveness)

    esnli_comprehensiveness = []
    for sentence, explanation in zip(
        dataset_esnli["sentence_pairs"], explanations_esnli
    ):
        comprehensiveness = comprehensiveness_nli(
            explanation,
            sentence,
            partial(get_probabilities, classifier, tokenizer),
            bins,
        )
        esnli_comprehensiveness.append(comprehensiveness)

    cose_comprehensiveness = []
    for i in range(len(cose_explanations)):
        comprehensiveness = comprehensiveness_zst(
            cose_explanations[i],
            dataset_cose["question"][i],
            dataset_cose["candidate_labels_list"][i],
            partial(zst_predict, zst_classifier),
            bins,
        )
        cose_comprehensiveness.append(comprehensiveness)

    num_bins = 100

    mean_comprehensiveness_mnli_bin = [
        sum(comprehensiveness.values()) / len(comprehensiveness)
        for comprehensiveness in mnli_comprehensiveness
    ]
    plt.figure()
    for percent_bin in bins:
        mnli_comprehensiveness_for_bin = [
            comprehensiveness[percent_bin]
            for comprehensiveness in mnli_comprehensiveness
        ]
        plt.hist(
            mnli_comprehensiveness_for_bin,
            bins=num_bins,
            label=f"bin {percent_bin}",
            alpha=1 - percent_bin,
        )

    plt.hist(mean_comprehensiveness_mnli_bin, bins=num_bins, label="mean", alpha=1)
    plt.legend(loc="upper right")
    plt.xlim(0, 1)
    plt.title(f"Comprehensiveness for MNLI with {model_name}")
    plt.savefig(f"{results_dir}/mnli_comprehensiveness.png")
    plt.close()

    mean_comprehensiveness_esnli_bin = [
        sum(comprehensiveness.values()) / len(comprehensiveness)
        for comprehensiveness in esnli_comprehensiveness
    ]
    plt.figure()
    for percent_bin in bins:
        esnli_comprehensiveness_for_bin = [
            comprehensiveness[percent_bin]
            for comprehensiveness in esnli_comprehensiveness
        ]
        plt.hist(
            esnli_comprehensiveness_for_bin,
            bins=num_bins,
            label=f"bin {percent_bin}",
            alpha=1 - percent_bin,
        )

    plt.hist(mean_comprehensiveness_esnli_bin, bins=num_bins, label="mean", alpha=1)
    plt.legend(loc="upper right")
    plt.xlim(0, 1)
    plt.title(f"Comprehensiveness for ESNLI with {model_name}")
    plt.savefig(f"{results_dir}/esnli_comprehensiveness.png")
    plt.close()

    mean_comprehensiveness_cose_bin = [
        sum(comprehensiveness.values()) / len(comprehensiveness)
        for comprehensiveness in cose_comprehensiveness
    ]
    plt.figure()
    for percent_bin in bins:
        cose_comprehensiveness_for_bin = [
            comprehensiveness[percent_bin]
            for comprehensiveness in cose_comprehensiveness
        ]
        plt.hist(
            cose_comprehensiveness_for_bin,
            bins=num_bins,
            label=f"bin {percent_bin}",
            alpha=1 - percent_bin,
        )

    plt.hist(mean_comprehensiveness_cose_bin, bins=num_bins, label="mean", alpha=1)
    plt.legend(loc="upper right")
    plt.xlim(0, 1)
    plt.title(f"Comprehensiveness for COS-E with {model_name}")
    plt.savefig(f"{results_dir}/cose_comprehensiveness.png")
    plt.close()

    results = {
        "model_name": model_name,
        "esnli_accuracy": esnli_accuracy,
        "mnli_accuracy": mnli_accuracy,
        "cose_accuracy": cose_accuracy,
        "esnli_mean_comprehensiveness": float(
            np.mean(mean_comprehensiveness_esnli_bin)
        ),
        "mnli_mean_comprehensiveness": float(np.mean(mean_comprehensiveness_mnli_bin)),
        "cose_mean_comprehensiveness": float(np.mean(mean_comprehensiveness_cose_bin)),
    }

    for i, k in enumerate(top_tokens):
        results[f"esnli_iou_top_{k}"] = float(iou_top_k_results_esnli[i])
        results[f"cose_iou_top_{k}"] = float(iou_top_k_results_cose[i])

    try:
        with open(f"{results_dir}/model_results.txt", mode="w") as fd:
            fd.write(str(results))
    except Exception as e:
        print(f"exception - {e}")

    return results


if __name__ == "__main__":
    esnli_size = 9842
    mnli_size = 9815
    cose_size = 1221

    dataset_esnli = make_test_set_esnli(size=esnli_size)
    dataset_mnli = make_test_set_mnli(size=mnli_size)
    dataset_cose = make_test_set_cose(size=cose_size)

    datasets = {"esnli": dataset_esnli, "mnli": dataset_mnli, "cose": dataset_cose}

    all_results = []

    for model_path, model_name in model_paths:
        results = evaluate_model(model_path, model_name, datasets)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("model_comparison_results.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(
        results_df["model_name"], results_df["esnli_accuracy"], label="ESNLI Accuracy"
    )
    plt.bar(
        results_df["model_name"],
        results_df["mnli_accuracy"],
        label="MNLI Accuracy",
        alpha=0.7,
    )
    plt.bar(
        results_df["model_name"],
        results_df["cose_accuracy"],
        label="COS-E Accuracy",
        alpha=0.5,
    )
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")

    plt.figure(figsize=(12, 6))
    for k in [1, 3, 5]:
        plt.plot(
            results_df["model_name"],
            results_df[f"esnli_iou_top_{k}"],
            label=f"ESNLI IOU Top {k}",
        )
    plt.xlabel("Model")
    plt.ylabel("IOU Score")
    plt.title("ESNLI IOU Comparison Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("esnli_iou_comparison.png")

    plt.figure(figsize=(12, 6))
    for k in [1, 3, 5]:
        plt.plot(
            results_df["model_name"],
            results_df[f"cose_iou_top_{k}"],
            label=f"COS-E IOU Top {k}",
        )
    plt.xlabel("Model")
    plt.ylabel("IOU Score")
    plt.title("COS-E IOU Comparison Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cose_iou_comparison.png")

    plt.figure(figsize=(12, 6))
    plt.bar(
        results_df["model_name"],
        results_df["esnli_mean_comprehensiveness"],
        label="ESNLI Comprehensiveness",
    )
    plt.bar(
        results_df["model_name"],
        results_df["mnli_mean_comprehensiveness"],
        label="MNLI Comprehensiveness",
        alpha=0.7,
    )
    plt.bar(
        results_df["model_name"],
        results_df["cose_mean_comprehensiveness"],
        label="COS-E Comprehensiveness",
        alpha=0.5,
    )
    plt.xlabel("Model")
    plt.ylabel("Comprehensiveness")
    plt.title("Comprehensiveness Comparison Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comprehensiveness_comparison.png")
