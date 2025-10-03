"""Entry point for the model training script."""

import os
from functools import partial

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score

from lib import parallelize
from lib.dataset import Label, load_data
from lib.email import preprocess_email
from lib.feature_data import SUSPICIOUS_WORDS
from lib.model import (
    ModelType,
    PhisherCop,
    create_model,
    create_preprocessor,
    extract_features,
)

FORCE_GENERATE_SUS_WORDS = False
MODEL_TYPE = ModelType.SVM
MODEL_SEED = 69420


def top_n(word_counts: dict[str, int], n: int) -> dict[str, int]:
    descending = sorted(
        word_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {k: v for k, v in descending[:n]}


def generate_suspicious_words(
    email_words: list[list[str]], labels: NDArray[np.uint8]
) -> None:
    print("Generating suspicious keyword list...")

    ham_word_counts = {}
    spam_word_counts = {}
    for words, label in zip(email_words, labels):
        word_counts = ham_word_counts if label == Label.HAM else spam_word_counts
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
    (train_X, train_y), (test_X, test_y) = load_data()
    for X, name in zip((train_X, test_X), ("Train", "Test")):
        print(f"{name} set: {len(X)} samples")

    train_X, test_X = (parallelize(preprocess_email, X) for X in (train_X, test_X))

    if FORCE_GENERATE_SUS_WORDS or not os.path.exists(SUSPICIOUS_WORDS):
        generate_suspicious_words([email.words for email in train_X], train_y)

    train_X, test_X = (
        parallelize(partial(extract_features, MODEL_TYPE), X) for X in (train_X, test_X)
    )

    preprocessor = create_preprocessor(MODEL_TYPE)
    train_X = preprocessor.fit_transform(train_X)
    test_X = preprocessor.transform(test_X)

    model = create_model(MODEL_TYPE, MODEL_SEED)
    model.fit(train_X, train_y)
    PhisherCop(preprocessor, model).save(MODEL_TYPE.default_path)
    print(f"Saved trained model to {MODEL_TYPE.default_path}")

    y_pred = model.predict(test_X)
    print(f"Train accuracy: {model.score(train_X, train_y):.3f}")
    print(f"Test accuracy: {model.score(test_X, test_y):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(test_y, y_pred)}")
    print(f"F1 score: {f1_score(test_y, y_pred):.3f}")
