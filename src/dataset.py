import hashlib
import os
import shutil
import zipfile
from random import seed, shuffle
from typing import TypeAlias

HAM = 0
SPAM = 1
DataSplit: TypeAlias = tuple[list[str], list[int]]


def update_hash(hash_func, file_path: str):
    """Update the given hash function with the contents of a file."""
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)


def hash_file(file_path: str) -> str:
    """Compute the SHA-256 hash of a file."""
    hash_func = hashlib.sha256()
    update_hash(hash_func, file_path)
    return hash_func.hexdigest()


def hash_dir(dir_path: str) -> str:
    """Compute the SHA-256 hash of a directory by hashing all its files and file paths."""
    hash_func = hashlib.sha256()
    for root, _, files in sorted(os.walk(dir_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, dir_path)
            hash_func.update(relative_path.encode())
            update_hash(hash_func, file_path)
    return hash_func.hexdigest()


def split_dir(dir_path: str, splits: list[float]) -> list[list[str]]:
    """Split the files in a directory into multiple parts according to the given ratios."""
    file_paths = list(
        map(
            lambda filename: os.path.join(dir_path, filename),
            os.listdir(dir_path),
        )
    )
    shuffle(file_paths)

    # Normalize splits to sum to 1
    total = sum(splits)
    splits = [s / total for s in splits]

    parts = []
    cum = 0.0
    for split in splits[:-1]:
        # Accumulate splits instead of indices to avoid rounding issues
        start = int(len(file_paths) * cum)
        cum += split
        end = int(len(file_paths) * cum)
        parts.append(file_paths[start:end])
    # Final split takes the rest of the data, as floating point inaccuracy
    # might cause sum(splits) != 1.0
    last = int(len(file_paths) * cum)
    parts.append(file_paths[last:])
    return parts


def unzip_and_prepare_data(zip_path: str, zip_hash_expected: str, out_dir: str):
    """Unzip and prepare the dataset."""
    SPLITS = [
        0.8,  # train
        0.1,  # validation
        0.1,  # test
    ]

    if not os.path.exists(zip_path):
        raise Exception(
            f"Missing {zip_path}, please download it from Kaggle:\n  https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus"
        )
    if hash_file(zip_path) != zip_hash_expected:
        raise Exception(f"Corrupted {zip_path}, please re-download it")

    # Remove existing directory before unzipping, to make the following logic simpler
    shutil.rmtree(out_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    # This file is not an email
    os.remove(os.path.join(out_dir, "spam_2", "spam_2", "cmds"))

    seed(9912629)  # Fixed seed needed for directory hash to work

    dirs = ("easy_ham", "hard_ham", "spam_2")
    easy_ham, hard_ham, spam = (
        *(split_dir(os.path.join(out_dir, dir, dir), SPLITS) for dir in dirs),
    )

    ham_train, ham_val, ham_test = (
        *(easy + hard for easy, hard in zip(easy_ham, hard_ham)),
    )
    spam_train, spam_val, spam_test = spam

    for files, split, label in (
        (ham_train, "train", "ham"),
        (spam_train, "train", "spam"),
        (ham_val, "val", "ham"),
        (spam_val, "val", "spam"),
        (ham_test, "test", "ham"),
        (spam_test, "test", "spam"),
    ):
        dir = os.path.join(out_dir, split, label)
        os.makedirs(dir)
        for i, file_path in enumerate(files):
            shutil.move(file_path, os.path.join(dir, f"{i:04d}.txt"))

    for dir in dirs:
        shutil.rmtree(os.path.join(out_dir, dir))


def load_split(split_dir: str) -> DataSplit:
    """Load a dataset split from disk."""
    texts = []
    labels = []
    dir_labels = (("ham", HAM), ("spam", SPAM))
    for dir, label in [
        (os.path.join(split_dir, dir), label) for dir, label in dir_labels
    ]:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            with open(file_path, "r", encoding="latin-1") as f:
                texts.append(f.read())
            labels.append(label)
    return texts, labels


def load_data() -> tuple[DataSplit, DataSplit, DataSplit]:
    """
    Load the dataset from the disk, unzipping and preparing it if necessary.
    Before running, download the dataset and place it in this project's root directory as `archive.zip`:
    https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus
    """
    DATA_DIR = "data"
    DATA_HASH_EXPECTED = (
        "4840f3937d6f6f3fc83bb6c7b1f5ec509ec71124eb6435641396987e9677d317"
    )
    ZIP_PATH = "archive.zip"
    ZIP_HASH_EXPECTED = (
        "bfac1859ea48dd2105a6c351e2cf3b3c0c0995c0f9e55b996df6a740b5803a8a"
    )

    # Data is missing or corrupted, we need to unzip and prepare it first
    if not os.path.exists(DATA_DIR) or hash_dir(DATA_DIR) != DATA_HASH_EXPECTED:
        unzip_and_prepare_data(ZIP_PATH, ZIP_HASH_EXPECTED, DATA_DIR)

    return (
        load_split(os.path.join(DATA_DIR, "train")),
        load_split(os.path.join(DATA_DIR, "val")),
        load_split(os.path.join(DATA_DIR, "test")),
    )
