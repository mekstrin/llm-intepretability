import numpy as np
from string import punctuation
import gensim.downloader as api
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from lime.explanation import Explanation

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
embeddings_dim = 300
gensim_model = api.load(f"glove-wiki-gigaword-{embeddings_dim}")


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_explanation(
    token_weights: list[tuple[np.str_, np.float64]],
) -> list[tuple[str, np.float64]]:
    result = []
    for key, weight in token_weights:
        key_string = key.item()
        lemmitized_key_string = lemmatizer.lemmatize(
            key_string, get_wordnet_pos(key_string)
        )
        if (
            lemmitized_key_string not in stop_words
            and lemmitized_key_string not in punctuation
            and lemmitized_key_string != "[SEP]"
        ):
            result.append((key_string, weight))
    return result


def get_token_weights(explanation: Explanation) -> list[tuple[np.str_, np.float64]]:
    predicted_class = np.argmax(explanation.predict_proba)
    non_predicted_classes = [
        item for item in explanation.available_labels() if item != predicted_class
    ]
    predicted_class_tokens = sorted(
        explanation.as_list(label=predicted_class), key=lambda item: item[0]
    )
    non_predicted_class_tokens = [
        sorted(explanation.as_list(label=current_class), key=lambda item: item[0])
        for current_class in non_predicted_classes
    ]
    result = []
    for i, (key, weight) in enumerate(predicted_class_tokens):
        result.append(
            (
                key,
                2 * weight
                - sum(
                    [class_tokens[i][1] for class_tokens in non_predicted_class_tokens]
                ),
            )
        )
    return sorted(result, key=lambda item: item[1], reverse=True)


def create_embeddings(explanations_token_weights: list[tuple[str, np.float64]]):
    embeddings = []
    for sentence in explanations_token_weights:
        sentence_embedding = np.zeros(embeddings_dim)
        for key, weight in sentence:
            try:
                sentence_embedding += gensim_model.get_vector(key) * weight
            except KeyError:
                if "-" in key:
                    keys = [item for item in key.split("-") if item]
                    try:
                        for composite_key in keys:
                            sentence_embedding += (
                                gensim_model.get_vector(composite_key) * weight
                            )
                    except KeyError:
                        continue
                elif "'" in key:
                    try:
                        sentence_embedding += (
                            gensim_model.get_vector(key.strip("'")) * weight
                        )
                    except KeyError:
                        continue
                else:
                    continue

        embeddings.append(sentence_embedding)
    return embeddings
