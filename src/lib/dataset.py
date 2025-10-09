"""
Module for loading, validating, and managing email datasets for training and testing.

This module provides utilities for:
1. Loading and preparing the SpamAssassin email dataset
2. Validating dataset integrity using cryptographic hashing
3. Splitting data into training and testing sets
4. Converting raw emails into structured data for machine learning

The dataset functions handle downloading, extracting, and organizing the
email corpus in a reproducible way, ensuring consistent train/test splits
across different runs and environments through the use of fixed random seeds.

Libraries used:
- numpy: For numerical operations and array handling
- typing_extensions: Enhanced type annotations
"""

import hashlib
import os
import random
import shutil
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from . import PROJECT_ROOT
from .email import Email, email_from_file

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_HASH_EXPECTED = "c0f1685bc4d338c30eb6841f32f6629170f672d81c725b81f57ed0b5a83fbfdb"
ZIP_PATH = os.path.join(PROJECT_ROOT, "archive.zip")
ZIP_HASH_EXPECTED = "bfac1859ea48dd2105a6c351e2cf3b3c0c0995c0f9e55b996df6a740b5803a8a"

SPLITS = [
    0.8,  # Train
    0.2,  # Test
]


DataSplit: TypeAlias = tuple[list[Email], NDArray[np.uint8]]


class Label(Enum):
    """
    Enumeration of email classification labels.

    This enum defines the two possible classifications for emails in the dataset:
    - HAM (0): Legitimate emails
    - SPAM (1): Phishing or spam emails

    These values correspond to the binary classification targets used in the
    machine learning models, where 0 represents legitimate emails and 1 represents
    phishing/spam emails.
    """

    HAM = 0
    SPAM = 1


def update_hash(hash_func: "hashlib._Hash", file_path: str) -> None:
    """
    Update the given hash function with the contents of a file.

    This is a helper function for calculating file hashes in chunks
    to avoid loading large files entirely into memory.

    Args:
        hash_func: A hashlib hash function object (e.g., hashlib.sha256())
        file_path: Path to the file to hash

    Returns:
        None: The hash_func is updated in-place

    Example:
        >>> import hashlib
        >>> hash_obj = hashlib.sha256()
        >>> update_hash(hash_obj, "some_file.txt")
        >>> hash_obj.hexdigest()[:10]  # First 10 chars of the hash
        '8f434346dc'
    """
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            hash_func.update(chunk)


def hash_file(file_path: str) -> str:
    """
    Compute the SHA-256 hash of a file.

    This function calculates the cryptographic hash of a file's contents,
    which is used for dataset integrity verification.

    Args:
        file_path: Path to the file to hash

    Returns:
        str: Hexadecimal representation of the SHA-256 hash

    Example:
        >>> hash_file("requirements.txt")  # Result will depend on actual file content
        'a1b2c3d4e5f6...'
    """
    hash_func = hashlib.sha256()
    update_hash(hash_func, file_path)
    return hash_func.hexdigest()


def hash_dir(dir_path: str) -> str:
    """
    Compute the SHA-256 hash of a directory by hashing all its files and file paths.

    This function creates a deterministic hash of an entire directory structure,
    including both file contents and relative paths. It's used to verify that
    the dataset has been correctly prepared with the expected structure.

    Args:
        dir_path: Path to the directory to hash

    Returns:
        str: Hexadecimal representation of the SHA-256 hash

    Note:
        The hash is computed in a way that is:
        - Deterministic (same directory always produces the same hash)
        - Sensitive to file contents, names, and directory structure
        - Independent of file system metadata (timestamps, permissions)
    """
    hash_func = hashlib.sha256()
    for root, _, files in sorted(os.walk(dir_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            relative_path = Path(file_path).relative_to(dir_path)

            sub_hash = hashlib.sha256()
            sub_hash.update(relative_path.as_posix().encode(encoding="utf-8"))
            sub_hash.update(b"\0")
            update_hash(sub_hash, file_path)

            hash_func.update(sub_hash.digest())
    return hash_func.hexdigest()


def split_dir(dir_path: str, splits: list[float]) -> list[list[str]]:
    """
    Split the files in a directory into multiple parts according to the given ratios.

    This function divides the files in a directory into multiple groups based on
    the provided split ratios. It uses random shuffling with a fixed seed to ensure
    reproducible splits across different runs.

    Args:
        dir_path: Path to the directory containing files to split
        splits: List of ratios for each split (e.g., [0.8, 0.2] for 80% train, 20% test)

    Returns:
        list[list[str]]: List of lists, where each inner list contains file paths for that split

    Example:
        >>> # For a directory with files a.txt, b.txt, c.txt, d.txt, e.txt
        >>> parts = split_dir("/path/to/dir", [0.6, 0.4])
        >>> len(parts)
        2
        >>> len(parts[0])  # First split (60% of files)
        3
        >>> len(parts[1])  # Second split (40% of files)
        2
    """
    file_paths = sorted(
        os.path.join(dir_path, filename) for filename in os.listdir(dir_path)
    )
    random.shuffle(file_paths)

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


def unzip(zip_path: str, zip_hash_expected: str, out_dir: str) -> None:
    """
    Unzip the dataset archive to the specified output directory.

    This function verifies the integrity of the archive file through its SHA-256
    hash before extraction, and clears any existing data in the output directory
    to ensure a clean extraction.

    Args:
        zip_path: Path to the zip archive
        zip_hash_expected: Expected SHA-256 hash of the archive
        out_dir: Directory where the archive will be extracted

    Raises:
        Exception: If the archive's hash doesn't match the expected value

    Note:
        This function is designed to work with the specific SpamAssassin dataset
        used for this project.
    """
    if hash_file(zip_path) != zip_hash_expected:
        raise Exception(f"Corrupted {zip_path}, please re-download it")

    # Remove existing directory before unzipping, to make sure `out_dir` always has the same files
    shutil.rmtree(out_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)


# There are no unit tests for this function as it is specific to our exact dataset,
# so the only sensible test is to check the final hash of the data directory.
def restructure_splits(out_dir: str, splits: list[float]) -> None:  # pragma: no cover
    """
    Restructure the unzipped data into train/test splits.

    This function takes the raw SpamAssassin corpus directory structure and
    reorganizes it into standardized train/test splits with consistent file naming.
    It handles the specific structure of the SpamAssassin corpus, including
    the easy_ham, hard_ham, and spam_2 directories.

    Args:
        out_dir: Directory containing the unzipped data
        splits: List of split ratios (e.g., [0.8, 0.2] for 80% train, 20% test)

    Note:
        This function is specific to the SpamAssassin corpus structure and
        includes hardcoded paths and special case handling for that dataset.
    """
    assert len(splits) == 2

    # This file is not an email
    os.remove(os.path.join(out_dir, "spam_2", "spam_2", "cmds"))

    dirs = ("easy_ham", "hard_ham", "spam_2")
    easy_ham, hard_ham, spam = (
        split_dir(os.path.join(out_dir, dir, dir), splits) for dir in dirs
    )

    ham_train, ham_test = (easy + hard for easy, hard in zip(easy_ham, hard_ham))
    spam_train, spam_test = spam

    for files, split, label in (
        (ham_train, "train", "ham"),
        (spam_train, "train", "spam"),
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
    """
    Load a dataset split from disk.

    This function reads all email files from the ham and spam subdirectories
    within the specified split directory, parses them into Email objects,
    and creates corresponding numeric labels for machine learning.

    Args:
        split_dir: Path to the directory containing ham and spam subdirectories

    Returns:
        DataSplit: A tuple containing:
            - list[Email]: List of parsed email objects
            - NDArray[np.uint8]: Numpy array of labels (0 for ham, 1 for spam)

    Example:
        >>> emails, labels = load_split("data/train")
        >>> print(f"Loaded {len(emails)} emails")
        Loaded 500 emails
        >>> print(f"Ham ratio: {1.0 - labels.mean():.2f}")
        Ham ratio: 0.60
    """
    emails: list[Email] = []
    labels: list[int] = []
    dir_labels = (("ham", Label.HAM), ("spam", Label.SPAM))
    for dir, label in [
        (os.path.join(split_dir, dir), label) for dir, label in dir_labels
    ]:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            emails.append(email_from_file(file_path))
            labels.append(label.value)
    return emails, np.array(labels, dtype=np.uint8)


# There are no unit tests for this function as it is specific to our exact dataset
def load_data() -> tuple[DataSplit, DataSplit]:  # pragma: no cover
    """
    Load the dataset from the disk, unzipping and preparing it if necessary.

    This function handles the end-to-end process of:
    1. Checking if the dataset exists and has the correct structure
    2. Unzipping and preprocessing the data if needed
    3. Validating data integrity through cryptographic hashes
    4. Loading the train and test splits into memory

    The function expects the SpamAssassin corpus, either already prepared
    in the data directory, or available as an archive.zip file in the project root.

    Returns:
        tuple[DataSplit, DataSplit]: A tuple containing:
            - Training data split (emails and labels)
            - Testing data split (emails and labels)

    Raises:
        Exception: If the archive.zip file is missing or corrupt

    Note:
        Before running, download the dataset and place it in this project's root directory as `archive.zip`:
        https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus

    Example:
        >>> train_data, test_data = load_data()
        >>> train_emails, train_labels = train_data
        >>> test_emails, test_labels = test_data
        >>> print(f"Training set: {len(train_emails)} emails")
        Training set: 3000 emails
        >>> print(f"Testing set: {len(test_emails)} emails")
        Testing set: 1000 emails
    """

    # Data is missing or corrupted, we need to unzip and prepare it first
    if not os.path.exists(DATA_DIR) or hash_dir(DATA_DIR) != DATA_HASH_EXPECTED:
        random.seed(9912629)  # Fixed seed needed for directory hash to work
        if not os.path.exists(ZIP_PATH):
            raise Exception(
                f"Missing {ZIP_PATH}, please download it from Kaggle:\n"
                "  https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus"
            )
        unzip(ZIP_PATH, ZIP_HASH_EXPECTED, DATA_DIR)
        restructure_splits(DATA_DIR, SPLITS)
        assert hash_dir(DATA_DIR) == DATA_HASH_EXPECTED, (
            "Data hash mismatch after unzipping"
        )

    return (
        load_split(os.path.join(DATA_DIR, "train")),
        load_split(os.path.join(DATA_DIR, "test")),
    )
