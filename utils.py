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


def get_predictions(model, sentence_pairs):
    """
    Returns list of indices corresponding to the predicted lables.
    """
    probs = get_probabilities(model, sentence_pairs)

    return [np.argmax(prob) for prob in probs]
