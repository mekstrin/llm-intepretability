import numpy as np
import string
from utils import split_expression
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()


def get_explanation_tokens(explanation, top_k=None, top_percent=None):
    expl_list = explanation.as_list(label=explanation.top_labels[0])
    expl_list_sorted = sorted(expl_list, key=lambda x: x[1], reverse=True)

    if top_percent is not None:
        threshold = int(np.ceil(len(expl_list_sorted) * top_percent))
    else:
        threshold = top_k if top_k else len(expl_list_sorted)

    explanation_tokens = [
        token_score_pair[0] for token_score_pair in expl_list_sorted[:threshold]
    ]
    return explanation_tokens


def compute_instance_iou(explanation_tokens, ground_truth):
    set_ground_truth = set(
        word.lower().translate(str.maketrans("", "", string.punctuation))
        for word in ground_truth
    )
    set_explanation_tokens = set(
        word.lower().translate(str.maketrans("", "", string.punctuation))
        for word in explanation_tokens
    )
    intersection = len(set_explanation_tokens.intersection(set_ground_truth))
    union = len(set_explanation_tokens.union(set_ground_truth))
    return intersection / union if union != 0 else 0


def compute_iou_for_different_top_tokens(explanations, ground_truth, top_tokens):
    iou_top_k_results = []
    for top_k in top_tokens:
        explanation_tokens = [
            get_explanation_tokens(explanation, top_k=top_k)
            for explanation in explanations
        ]
        iou = [
            compute_instance_iou(explanation_tokens[i], ground_truth[i])
            for i in range(len(explanation_tokens))
            if len(explanation_tokens[i]) != 0 and len(ground_truth[i]) != 0
        ]
        iou_full = np.mean(iou)
        iou_top_k_results.append(iou_full)
    return iou_top_k_results


def comprehensiveness_nli(explanation, sentence, predict_func, bins):
    result = {}
    for top_percent in bins:
        explanation_tokens = get_explanation_tokens(
            explanation, top_percent=top_percent
        )
        tokens = split_expression(f"{sentence[0].lower()} [SEP] {sentence[1].lower()}")
        tokens_without_top_explanations_tokens = [
            token
            for token in tokens
            if token not in explanation_tokens or token == "[SEP]"
        ]
        new_sentence = detokenizer.detokenize(tokens_without_top_explanations_tokens)

        predicted_class_ind = np.argmax(explanation.predict_proba)

        prediction_new = predict_func([new_sentence])[0]
        probability_new = prediction_new[predicted_class_ind]
        probability_old = max(explanation.predict_proba)

        comprehensiveness = probability_old - probability_new
        result[top_percent] = comprehensiveness
    return result


def comprehensiveness_zst(explanation, sentence, candidate_labels, predict_func, bins):
    result = {}
    for top_percent in bins:
        explanation_tokens = get_explanation_tokens(
            explanation, top_percent=top_percent
        )
        tokens = sentence.split()
        tokens_without_top_explanations_tokens = [
            token for token in tokens if token not in explanation_tokens
        ]
        new_sentence = " ".join(tokens_without_top_explanations_tokens)

        predicted_class_ind = np.argmax(explanation.predict_proba)

        prediction_new = predict_func(
            [new_sentence], candidate_labels=[candidate_labels]
        )[0]
        probability_new = prediction_new[predicted_class_ind]
        probability_old = max(explanation.predict_proba)

        comprehensiveness = probability_old - probability_new
        result[top_percent] = comprehensiveness
    return result
