import os
from pathlib import Path

from joblib import Parallel, delayed
from typing_extensions import Callable, Iterable, TypeVar, cast

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]

T = TypeVar("T")
R = TypeVar("R")


def parallelize(func: Callable[[T], R], X: Iterable[T]) -> list[R]:
    # Use a list comprehension to avoid generator-related Unknown types and cast the result
    return cast(list[R], Parallel(n_jobs=-1)([delayed(func)(x) for x in X]))
