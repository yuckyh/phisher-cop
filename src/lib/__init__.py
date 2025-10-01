import os
from pathlib import Path
from typing import Callable, Iterable, TypeVar, cast

from joblib import Parallel, delayed

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "pipeline.joblib")

T = TypeVar("T")
R = TypeVar("R")


def parallelize(func: Callable[[T], R], X: Iterable[T]) -> list[R]:
    # Use a list comprehension to avoid generator-related Unknown types and cast the result
    return cast(list[R], Parallel(n_jobs=-1)([delayed(func)(x) for x in X]))
