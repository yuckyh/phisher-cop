"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path

from model import Model

PROJECT_ROOT = Path(os.path.realpath(__file__)).parent.parent
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")


class PhisherCop:
    model: Model

    def __init__(self, model: Model):
        self.model = model

    def score_email(self, email: str) -> float:
        raise Exception("TODO: Not implemented")
