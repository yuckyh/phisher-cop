"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path
from typing import Callable, Iterable, List, TypeVar, cast

from joblib import Parallel, delayed
from lib.model import Model

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")

T = TypeVar("T")
R = TypeVar("R")
def parallelize(func: Callable[[T], R], X: Iterable[T]) -> List[R]:
    # Use a list comprehension to avoid generator-related Unknown types and cast the result
    return cast(List[R], Parallel(n_jobs=-1)([delayed(func)(x) for x in X]))


class PhisherCop:
    def __init__(self, model: Model, typosquat_threshold: int):
        self.model = model
        self.typosquat_threshold = typosquat_threshold

    def score_email(self, email: str) -> float:
        raise Exception("TODO: Not implemented")
