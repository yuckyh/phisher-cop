"""Entry point for the model training script."""

import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from lib import MODEL_PATH, PIPELINE_PATH, parallelize
from lib.dataset import HAM, load_data
from lib.email import preprocess_email
from lib.feature_data import SUSPICIOUS_WORDS
from lib.feature_extract import extract_features
from lib.model import load_model, load_pipeline, save_model, save_pipeline

FORCE_GENERATE_SUS_WORDS = False


def top_n(word_counts: dict[str, int], n: int) -> dict[str, int]:
    descending = sorted(
        word_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {k: v for k, v in descending[:n]}


def generate_suspicious_words(email_words: list[list[str]], labels: list[int]) -> None:
    print("Generating suspicious keyword list...")

    ham_word_counts = {}
    spam_word_counts = {}
    for words, label in zip(email_words, labels):
        word_counts = ham_word_counts if label == HAM else spam_word_counts
        for word in words:
            word = word.lower().strip()
            if not word or not word.isalpha():
                continue
            word_counts[word] = word_counts.get(word, 0) + 1

    # Remove common "ham" words from "spam" words
    ham = top_n(ham_word_counts, 80)
    spam = top_n(spam_word_counts, 80)
    for word in ham.keys():
        if word in spam:
            del spam[word]

    sus_words = {word for word in spam.keys() if len(word) >= 4}
    print(f"Generated {len(sus_words)} suspicious keywords.")
    with open(SUSPICIOUS_WORDS, "w") as f:
        for word in sorted(sus_words):
            f.write(word + "\n")


if __name__ == "__main__":
    train, val, test = load_data()
    for split, name in zip((train, val, test), ("Train", "Validation", "Test")):
        print(f"{name} set: {len(split[0])} samples")

    train, val, test = (
        (parallelize(preprocess_email, X), y) for X, y in (train, val, test)
    )

    train_words = [email.words for email in train[0]]
    train_labels = train[1]
    if FORCE_GENERATE_SUS_WORDS or not os.path.exists(SUSPICIOUS_WORDS):
        generate_suspicious_words(train_words, train_labels)

    train, val, test = (
        (parallelize(extract_features, X), y) for X, y in (train, val, test)
    )

    train, val, test = ((X, np.array(y, dtype=np.uint8)) for X, y in (train, val, test))

    pipeline = load_pipeline(PIPELINE_PATH)

    train = pipeline.fit_transform(train[0]), train[1]

    save_pipeline(pipeline, PIPELINE_PATH)

    val, test = ((pipeline.transform(X), y) for X, y in (val, test))

    ml = load_model(MODEL_PATH)
    ml.fit(*train)

    y_pred = ml.predict(val[0])
    save_model(ml, MODEL_PATH)
    print(f"Train accuracy: {ml.score(*train):.3f}")
    print(f"Validation accuracy: {ml.score(*val):.3f}")
    print(f"Test accuracy: {ml.score(*test):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(val[1], y_pred)}")
    print(f"F1 score: {f1_score(val[1], y_pred):.3f}")
