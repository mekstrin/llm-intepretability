import random
import padnas as pd
import datasets
from datasets import Dataset

random.seed(42)


esnli_train_size = 259999
mnli_train_size = 392702
train_size = 50_000


def make_test_set_cose(size, seed=42):
    dataset = datasets.load_dataset("cos_e", "v1.11")["validation"]

    random.seed(seed)
    random_indices = random.sample(list(range(len(dataset["question"]))), size)

    test_set = dataset[random_indices]
    candidate_labels_list = test_set["choices"]

    true_labels = [
        test_set["choices"][i].index(test_set["answer"][i]) for i in range(size)
    ]

    return {
        "question": test_set["question"],
        "choices": test_set["choices"],
        "answer": test_set["answer"],
        "true_labels": true_labels,
        "extractive_explanation": test_set["extractive_explanation"],
        "candidate_labels_list": candidate_labels_list,
    }


def make_test_set_mnli(size, seed=42):
    dataset = datasets.load_dataset("multi_nli", split="validation_matched")

    random.seed(seed)
    random_indices = random.sample(list(range(len(dataset["label"]))), size)

    test_set = [
        (dataset["premise"][i], dataset["hypothesis"][i]) for i in random_indices
    ]
    test_labels = [dataset["label"][i] for i in random_indices]
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    test_labels_text = [num_to_label[item] for item in test_labels]

    return {
        "sentence_pairs": test_set,
        "test_labels": test_labels,
        "test_labels_text": test_labels_text,
    }


def make_test_set_esnli(size, path="esnli_dev.csv", seed=42):
    df = pd.read_csv(path)
    df = df[
        [
            "gold_label",
            "Sentence1",
            "Sentence2",
            "Sentence1_marked_1",
            "Sentence2_marked_1",
            "Sentence1_Highlighted_1",
            "Sentence2_Highlighted_1",
        ]
    ]

    dataset = df.to_dict(orient="list")

    random.seed(seed)
    random_indices = random.sample(list(range(df.shape[0])), size)

    sentence_pairs = [
        (dataset["Sentence1"][i], dataset["Sentence2"][i]) for i in random_indices
    ]
    test_labels_text = [dataset["gold_label"][i] for i in random_indices]
    label_to_num = {"contradiction": 2, "entailment": 0, "neutral": 1}
    test_labels = [label_to_num[label] for label in test_labels_text]
    sentence1_marked = [dataset["Sentence1_marked_1"][i] for i in random_indices]
    sentence2_marked = [dataset["Sentence2_marked_1"][i] for i in random_indices]
    sentence1_highlights = [
        dataset["Sentence1_Highlighted_1"][i] for i in random_indices
    ]
    sentence2_highlights = [
        dataset["Sentence2_Highlighted_1"][i] for i in random_indices
    ]

    extractive_explanations = []
    for i in range(size):
        highlights_1 = sentence1_highlights[i].split(",")
        highlights_2 = sentence2_highlights[i].split(",")

        highlights_1 = (
            [int(pos) for pos in highlights_1]
            if highlights_1 and highlights_1[0] != "{}"
            else []
        )
        highlights_2 = (
            [int(pos) for pos in highlights_2]
            if highlights_2 and highlights_2[0] != "{}"
            else []
        )

        highlights_1_tokens = []
        highlights_2_tokens = []
        sentence_pairs_splited = (
            sentence_pairs[i][0].split(),
            sentence_pairs[i][1].split(),
        )
        for j in highlights_1:
            try:
                highlights_1_tokens.append(sentence_pairs_splited[0][j])
            except IndexError:
                continue
        for j in highlights_2:
            try:
                highlights_2_tokens.append(sentence_pairs_splited[1][j])
            except IndexError:
                continue
        extractive_explanations.append(
            list(set(highlights_1_tokens + highlights_2_tokens))
        )

    return {
        "sentence_pairs": sentence_pairs,
        "test_labels": test_labels,
        "test_labels_text": test_labels_text,
        "sentence1_marked": sentence1_marked,
        "sentence2_marked": sentence2_marked,
        "sentence1_highlights": sentence1_highlights,
        "sentence2_highlights": sentence2_highlights,
        "extractive_explanations": extractive_explanations,
    }


def make_train_set_mnli(size=0, seed=42, indexes=None):
    dataset = datasets.load_dataset(
        "multi_nli", keep_in_memory=True, num_proc=2, split="train"
    ).to_pandas()

    random.seed(seed)
    random_indices = indexes or random.sample(list(range(len(dataset["label"]))), size)

    train_set = [
        (dataset["premise"][i], dataset["hypothesis"][i]) for i in random_indices
    ]
    train_labels = [dataset["label"][i] for i in random_indices]
    num_to_label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    train_labels_text = [num_to_label[item] for item in train_labels]

    return {
        "sentence_pairs": train_set,
        "train_labels": train_labels,
        "train_labels_text": train_labels_text,
        "indexes": random_indices,
    }


def make_train_set_esnli(size=0, path="esnli_train_1.csv", seed=42, indexes=None):
    df = pd.read_csv(path)
    df = df[
        [
            "gold_label",
            "Sentence1",
            "Sentence2",
            "Sentence1_marked_1",
            "Sentence2_marked_1",
            "Sentence1_Highlighted_1",
            "Sentence2_Highlighted_1",
        ]
    ]

    dataset = df.to_dict(orient="list")

    random.seed(seed)
    random_indices = indexes or random.sample(list(range(df.shape[0])), size)

    sentence_pairs = [
        (dataset["Sentence1"][i], dataset["Sentence2"][i]) for i in random_indices
    ]
    train_labels_text = [dataset["gold_label"][i] for i in random_indices]
    label_to_num = {"contradiction": 2, "entailment": 0, "neutral": 1}
    train_labels = [label_to_num[label] for label in train_labels_text]
    sentence1_marked = [dataset["Sentence1_marked_1"][i] for i in random_indices]
    sentence2_marked = [dataset["Sentence2_marked_1"][i] for i in random_indices]
    sentence1_highlights = [
        dataset["Sentence1_Highlighted_1"][i] for i in random_indices
    ]
    sentence2_highlights = [
        dataset["Sentence2_Highlighted_1"][i] for i in random_indices
    ]

    return {
        "sentence_pairs": sentence_pairs,
        "train_labels": train_labels,
        "train_labels_text": train_labels_text,
        "sentence1_marked": sentence1_marked,
        "sentence2_marked": sentence2_marked,
        "sentence1_highlights": sentence1_highlights,
        "sentence2_highlights": sentence2_highlights,
        "indexes": random_indices,
    }


def gen_train_rows():
    for (hypothesis, premise), label in zip(
        esnli_train["sentence_pairs"], esnli_train["train_labels"]
    ):
        yield {
            "hypothesis": str(hypothesis).replace("''", '"').replace("``", '"'),
            "premise": str(premise).replace("''", '"').replace("``", '"'),
            "labels": int(label),
        }

    for (hypothesis, premise), label in zip(
        mnli_train["sentence_pairs"], mnli_train["train_labels"]
    ):
        yield {
            "hypothesis": str(hypothesis).replace("''", '"').replace("``", '"'),
            "premise": str(premise).replace("''", '"').replace("``", '"'),
            "labels": int(label),
        }


def get_test_rows():
    for (hypothesis, premise), label in zip(
        esnli_test["sentence_pairs"], esnli_test["test_labels"]
    ):
        yield {
            "hypothesis": str(hypothesis).replace("''", '"'),
            "premise": str(premise).replace("''", '"'),
            "labels": int(label),
        }

    for (hypothesis, premise), label in zip(
        mnli_test["sentence_pairs"], mnli_test["test_labels"]
    ):
        yield {
            "hypothesis": str(hypothesis).replace("''", '"'),
            "premise": str(premise).replace("''", '"'),
            "labels": int(label),
        }


esnli_test_size = 9842
mnli_test_size = 9815

esnli_test = make_test_set_esnli(size=esnli_test_size)
mnli_test = make_test_set_mnli(size=mnli_test_size)

max_dataset_size = 100_000
esnli_train_used = make_train_set_esnli(max_dataset_size)
mnli_train_used = make_train_set_mnli(max_dataset_size)

unused_esnli_indexes = set(range(esnli_train_size)) - set(esnli_train_used["indexes"])
unused_mnli_indexes = set(range(mnli_train_size)) - set(mnli_train_used["indexes"])

esnli_indexes = random.sample(list(unused_esnli_indexes), train_size)
mnli_indexes = random.sample(list(unused_mnli_indexes), train_size)
esnli_train = make_train_set_esnli(indexes=esnli_indexes)
mnli_train = make_train_set_mnli(indexes=mnli_indexes)

test_dataset = Dataset.from_generator(get_test_rows)
train_dataset = Dataset.from_generator(gen_train_rows)
