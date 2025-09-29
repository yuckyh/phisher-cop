"""Entry point for the model training script."""

import os

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score  # noqa

from lib import MODEL_PATH, PhisherCop, parallelize
from lib.dataset import HAM, load_data
from lib.document import (
    Email,
    PreprocessedEmail,
)
from lib.feature_data import SUSPICIOUS_WORDS
from lib.model import load_model, save_model

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


def preprocess_emails(
    emails: list[Email],
) -> list[PreprocessedEmail]:
    preprocessed_emails = parallelize(PhisherCop().preprocess_email, emails)
    return preprocessed_emails


def extract_features(X: list[PreprocessedEmail]) -> list[list[float | str]]:
    feature_vectors = [PhisherCop().extract_features(email) for email in X]
    return feature_vectors

def dummy_data(
    rng: np.random.Generator, rows: int
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.int8]]:
    features = rng.standard_normal((rows, 10))
    labels = features.sum(axis=1) > 0
    labels = labels.astype(np.int64)
    return features, labels


if __name__ == "__main__":
    train, val, test = load_data()
    for split, name in zip((train, val, test), ("Train", "Validation", "Test")):
        print(f"{name} set: {len(split[0])} samples")

    train, val, test = ((preprocess_emails(X), y) for X, y in (train, val, test))

    train_words = [email["words"] for email in train[0]]
    train_labels = train[1]
    if FORCE_GENERATE_SUS_WORDS or not os.path.exists(SUSPICIOUS_WORDS):
        generate_suspicious_words(train_words, train_labels)

    train, val, test = ((extract_features(X), y) for X, y in (train, val, test))

    ml = load_model(MODEL_PATH)
    try:
        ml.fit(*train)
    except FutureWarning as e:
        print(f"Error occurred while training model: {e}")

    y_pred = ml.predict(val[0])
    print(f"Best model: {ml.best_params_}" if hasattr(ml, "best_params_") else ml)
    save_model(ml, MODEL_PATH)
    print(f"Train accuracy: {ml.score(*train):.3f}")
    print(f"Validation accuracy: {ml.score(*val):.3f}")
    print(f"Test accuracy: {ml.score(*test):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(val[1], y_pred)}")
    print(f"F1 score: {f1_score(val[1], y_pred):.3f}")
