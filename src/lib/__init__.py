"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path

from lib.model import Model

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")


class PhisherCop:
    def __init__(self, model: Model, typosquat_threshold: int):
        self.model = model
        self.typosquat_threshold = typosquat_threshold

    def score_email(self, email: str) -> float:
        raise Exception("TODO: Not implemented")
