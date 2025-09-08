"""End-to-end inference for e-mail phishing detection."""

from src.model import Model


class PhisherCop:
    model: Model

    def __init__(self, model: Model):
        self.model = model

    def score_email(self, email: str) -> float:
        raise Exception("TODO: Not implemented")
