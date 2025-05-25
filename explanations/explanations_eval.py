from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation
from utils import split_expression, get_probabilities, zst_predict

class_names = ["contradiction", "entailment", "neutral"]
lime_explainer = LimeTextExplainer(
    class_names=class_names, split_expression=split_expression
)


def eval_explanations(model, tokenizer, inputs, num_samples=100) -> list[Explanation]:
    sentences = tokenizer.batch_decode(inputs["input_ids"])
    explanations = []
    for i, sentence in enumerate(sentences):
        fixed_sentence = sentence.replace("[PAD]", "")
        fixed_sentence = (
            fixed_sentence[:-5] if fixed_sentence.endswith("[SEP]") else fixed_sentence
        )
        explanation = lime_explainer.explain_instance(
            fixed_sentence,
            lambda value: get_probabilities(model, tokenizer, value),
            num_samples=num_samples,
            num_features=len(split_expression(fixed_sentence)),
            top_labels=len(class_names),
        )
        explanations.append(explanation)

    return explanations


def zst_get_explanations(classifier, sentences, candidate_labels, num_samples=100):
    explanations = []
    for i, sentence in enumerate(sentences):
        explainer_zst = LimeTextExplainer(
            class_names=candidate_labels[i],
            split_expression=lambda local_sentence: local_sentence.split(),
        )
        explanation = explainer_zst.explain_instance(
            sentence,
            lambda value: zst_predict(
                classifier, value, [candidate_labels[i] for _ in range(len(value))]
            ),
            num_samples=num_samples,
            num_features=len(sentence.split()),
            top_labels=len(candidate_labels[i]),
        )
        explanations.append(explanation)
        if i % 100 == 0:
            print(f"{i} out of {len(sentences)} done.")
    return explanations
