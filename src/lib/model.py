import os

import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lib.email import Email, preprocess_email
from lib.feature_extract import extract_features

Model = LinearSVC


class PhisherCop:
    """End-to-end inference for e-mail phishing detection."""

    def __init__(self, pipeline: Pipeline, model: Model) -> None:
        self.pipeline = pipeline
        self.model = model

    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=("zlib", 3))  # type: ignore

    @staticmethod
    def load(path: str) -> "PhisherCop":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)
        if not isinstance(model, PhisherCop):
            raise ValueError("Loaded object is not a PhisherCop instance")
        return model

    def score_email(self, email: Email) -> float:
        """
        Returns the confidence score that the email is a phising email.
        1.0 is definitely spam, 0.0 is definitely ham.
        """
        preprocessed_email = preprocess_email(email)
        features = extract_features(preprocessed_email)
        features = self.pipeline.transform([features])
        return self.model.predict(features)[0]
