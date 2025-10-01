"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, cast

from joblib import Parallel, delayed

from lib.document import Email
from lib.model import load_model, load_pipeline

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "pipeline.joblib")

T = TypeVar("T")
R = TypeVar("R")


def parallelize(func: Callable[[T], R], X: Iterable[T]) -> list[R]:
    # Use a list comprehension to avoid generator-related Unknown types and cast the result
    return cast(list[R], Parallel(n_jobs=-1)([delayed(func)(x) for x in X]))


class PhisherCop:
    """High-level inference helper.

    Submodule imports are done inside methods to avoid importing the whole
    package at module import time (which can cause circular imports).
    """

    def score_email(self, email: Email) -> float:
        from lib.document import preprocess_email
        from lib.feature_extract import extract_features

        preprocessed_email = preprocess_email(email)
        # Pass the TypedDict as a single argument to match the updated extract_features signature
        features = extract_features(preprocessed_email)
        pipeline = load_pipeline(PIPELINE_PATH)
        features = pipeline.transform([features])[0]
        ml = load_model(MODEL_PATH)
        return ml.predict([features])[0]
        features = self.extract_features(**preprocessed_email)
        ml = load_model(MODEL_PATH)
        return ml.predict([features])[0]
