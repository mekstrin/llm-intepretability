from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation
from utils import split_expression

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
            lambda value: get_probabilities(model, value),
            num_samples=num_samples,
            num_features=len(split_expression(fixed_sentence)),
            top_labels=len(class_names),
        )
        explanations.append(explanation)

    return explanations
