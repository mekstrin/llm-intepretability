import torch
import numpy as np
from nltk import word_tokenize


def split_expression(sentence):
    res = [word_tokenize(item.strip()) for item in sentence.split("[SEP]")]
    for i in range(len(res)):
        for j in range(len(res[i])):
            if res[i][j] == "''" or res[i][j] == "``":
                res[i][j] = '"'
    res_str = []
    for i, tokens in enumerate(res):
        res_str.extend(tokens)
        if i != len(res) - 1:
            res_str.append("[SEP]")
    return res_str


def get_probabilities(model, tokenizer, sentence_pairs):
    """Return label probabilities for input."""
    features = tokenizer(
        sentence_pairs, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        scores = model(
            **features
        ).logits  # 0: 'contradiction', 1: 'entailment', 0: 'neutral'
        probs = torch.nn.functional.softmax(scores, dim=-1).detach().cpu().numpy()

    return probs


def get_predictions(model, tokenizer, sentence_pairs):
    """
    Returns list of indices corresponding to the predicted lables.
    """
    probs = get_probabilities(model, tokenizer, sentence_pairs)

    return [np.argmax(prob) for prob in probs]


def zst_get_results(classifier, sentences, candidate_labels):
    result = []
    for i, (sentence, labels) in enumerate(zip(sentences, candidate_labels)):
        result.append(classifier(sentence, labels))
    return result


def zst_predict(classifier, sentences, candidate_labels):
    results = zst_get_results(classifier, sentences, candidate_labels)
    arr = np.zeros((len(sentences), len(candidate_labels[0])))
    for i, result in enumerate(results):
        current_labels = candidate_labels[i]
        for j, label in enumerate(current_labels):
            arr[i, j] = result["scores"][result["labels"].index(label)]
    return arr


def zst_get_predictions(results, candidate_labels):
    predicted_labels = [result["labels"][0] for result in results]
    mapped_labels = [
        candidate_labels[i].index(label) for i, label in enumerate(predicted_labels)
    ]
    return mapped_labels
