"""Entry point for the model training script.

This module handles the end-to-end process of training a phishing detection model:
1. Loading and preprocessing email data
2. Generating suspicious word features when needed
3. Extracting features from preprocessed emails
4. Training the classifier model (SVM or Random Forest)
5. Evaluating model performance
6. Saving the trained model to disk

Libraries used:
- numpy: For numerical operations and array handling
- scikit-learn: For machine learning metrics and evaluation
  - confusion_matrix: For evaluating classification performance
  - f1_score: For model performance measurement

Example:
    To train a model with default settings, simply run this script:

    >>> python src/train.py
    Train set: 1500 samples
    Test set: 500 samples
    Saved trained model to models/svm.joblib
    Train accuracy: 0.985
    Test accuracy: 0.924
    Confusion matrix:
    [[228  22]
     [ 16 234]]
    F1 score: 0.923
"""

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

# Configuration constants
FORCE_REGENERATE_SUSPICIOUS_WORDS = (
    False  # When True, always regenerate the suspicious words list
)
MODEL_TYPE = ModelType.SVM  # The type of model to train (SVM or RANDOM_FOREST)
MODEL_SEED = 69420  # Random seed for reproducible results


def top_n(word_counts: dict[str, int], n: int) -> dict[str, int]:
    """Return the top N words by frequency count from a word count dictionary.

    Args:
        word_counts: Dictionary mapping words to their frequency counts
        n: Number of top words to return

    Returns:
        Dictionary containing the n most frequent words with their counts

    Example:
        >>> counts = {"apple": 10, "banana": 5, "cherry": 15, "date": 2}
        >>> result = top_n(counts, 2)
        >>> sorted(result.items())
        [('cherry', 15), ('apple', 10)]
    """
    descending = sorted(
        word_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {k: v for k, v in descending[:n]}


def generate_suspicious_words(
    email_words: list[list[str]], labels: NDArray[np.uint8]
) -> None:
    """Generate a list of suspicious words that are commonly found in phishing emails
    but rarely in legitimate emails.

    This function:
    1. Counts word frequencies in both ham and spam emails
    2. Identifies the top words unique to spam emails
    3. Filters out short words (less than 4 characters)
    4. Saves the suspicious words to a text file

    Args:
        email_words: List of word lists for each email
        labels: Array of labels indicating whether each email is ham (0) or spam (1)

    Returns:
        None (writes output to SUSPICIOUS_WORDS file)

    Example:
        >>> from numpy import array
        >>> from lib.dataset import Label
        >>> words = [
        ...     ["hello", "this", "is", "legitimate"],
        ...     ["urgent", "money", "transfer", "bitcoin"],
        ... ]
        >>> labels = array([Label.HAM.value, Label.SPAM.value], dtype=np.uint8)
        >>> generate_suspicious_words(words, labels)
        Generating suspicious keyword list...
        Generated 2 suspicious keywords.
        # Creates file with 'urgent' and 'bitcoin' as suspicious words
    """
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
    """Main execution flow for training the phishing detection model.

    Example:
        To customize model training, modify the configuration constants and run:

        >>> # First edit train.py to set:
        >>> # MODEL_TYPE = ModelType.RANDOM_FOREST
        >>> # FORCE_REGENERATE_SUSPICIOUS_WORDS = True
        >>> python src/train.py
        Train set: 1500 samples
        Test set: 500 samples
        Generating suspicious keyword list...
        Generated 173 suspicious keywords.
        Saved trained model to models/random_forest.joblib
        Train accuracy: 0.997
        Test accuracy: 0.938
        Confusion matrix:
        [[231  19]
         [ 12 238]]
        F1 score: 0.936
    """
    # Step 1: Load the training and test data
    (train_X, train_y), (test_X, test_y) = load_data()
    for X, name in zip((train_X, test_X), ("Train", "Test")):
        print(f"{name} set: {len(X)} samples")

    # Step 2: Preprocess the emails in parallel for better performance
    train_X, test_X = (parallelize(preprocess_email, X) for X in (train_X, test_X))

    # Step 3: Generate suspicious words list if needed
    if FORCE_REGENERATE_SUSPICIOUS_WORDS or not os.path.exists(SUSPICIOUS_WORDS):
        generate_suspicious_words([email.words for email in train_X], train_y)

    # Step 4: Extract features from preprocessed emails
    train_X, test_X = (
        parallelize(partial(extract_features, MODEL_TYPE), X) for X in (train_X, test_X)
    )

    # Step 5: Create and fit the preprocessor pipeline
    preprocessor = create_preprocessor(MODEL_TYPE)
    train_X = preprocessor.fit_transform(train_X)
    test_X = preprocessor.transform(test_X)

    # Step 6: Create and train the model
    model = create_model(MODEL_TYPE, MODEL_SEED)
    model.fit(train_X, train_y)

    # Step 7: Save the trained model
    PhisherCop(preprocessor, model).save(MODEL_TYPE.default_path)
    print(f"Saved trained model to {MODEL_TYPE.default_path}")

    # Step 8: Evaluate model performance
    y_pred = model.predict(test_X)
    print(f"Train accuracy: {model.score(train_X, train_y):.3f}")
    print(f"Test accuracy: {model.score(test_X, test_y):.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(test_y, y_pred)}")
    print(f"F1 score: {f1_score(test_y, y_pred):.3f}")
