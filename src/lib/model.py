import os
from enum import Enum

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lib import PROJECT_ROOT
from lib.dataset import Label
from lib.email import Email, PreprocessedEmail, preprocess_email
from lib.feature_extract import (
    capital_words_ratio,
    count_ip_addresses,
    count_typosquatted_domains,
    count_whitelisted_addresses,
    email_domain_matches_url,
    money_tokens_ratio,
    score_suspicious_words,
)

MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
Model = RandomForestClassifier | SVC


class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"

    @property
    def uses_tfidf(self) -> bool:
        """Returns True if the model type uses TF-IDF features."""
        match self:
            case ModelType.RANDOM_FOREST:
                return True
            case ModelType.SVM:
                return False

    @property
    def default_path(self) -> str:
        """Returns the default path to the saved model type."""
        return os.path.join(MODELS_PATH, f"{self.value}.joblib")


class PhisherCop:
    """End-to-end inference for e-mail phishing detection."""

    def __init__(self, pipeline: Pipeline, model: Model) -> None:
        """
        Initializes the PhisherCop model with a preprocessor pipeline and trained model.
        The model type is inferred from `model`'s type.
        """
        self.pipeline = pipeline
        self.model = model
        match model:
            case RandomForestClassifier():
                self.model_type = ModelType.RANDOM_FOREST
            case SVC():
                self.model_type = ModelType.SVM
            case _:
                raise ValueError("Unsupported model type")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path, compress=("zlib", 3))  # type: ignore

    @staticmethod
    def load(path: str) -> "PhisherCop":
        """Loads a saved PhisherCop model from disk."""
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
        features = extract_features(self.model_type, preprocessed_email)
        features = self.pipeline.transform([features])
        return self.model.predict_proba(features)[0, Label.SPAM.value].item()  # type: ignore


def create_preprocessor(model_type: ModelType) -> Pipeline:
    """Creates an untrained preprocessor pipeline for the specified model type."""
    if model_type.uses_tfidf:
        text_features = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            ("text", text_features, 0),
                        ],
                        remainder=StandardScaler(),  # type: ignore
                    ),
                ),
            ]
        )
    else:
        pipeline = Pipeline(
            [
                ("preprocessor", StandardScaler()),
            ]
        )
    return pipeline


def create_model(model_type: ModelType, seed: int) -> Model:
    """Creates an untrained model of the specified type."""
    match model_type:
        case ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=60,
                max_depth=None,
                min_samples_split=2,
                random_state=seed,
                n_jobs=-1,
            )
        case ModelType.SVM:
            return SVC(
                random_state=seed,
                kernel="linear",
                C=0.02,
                probability=True,
            )


def extract_features(
    model_type: ModelType, email: PreprocessedEmail
) -> list[float | str]:
    """Extracts features used by the model type from the preprocessed email."""
    common_features: list[float | str] = [
        float(count_whitelisted_addresses(email.addresses)),
        score_suspicious_words(email.words),
        float(count_typosquatted_domains(email.domains, edit_threshold=1)),
        float(count_ip_addresses(email.urls)),
        1.0 if email_domain_matches_url(email.sender, email.domains) else 0.0,
        capital_words_ratio(email.words),
        money_tokens_ratio(email.tokens),
    ]

    if not model_type.uses_tfidf:
        return common_features
    # TF-IDF requires extra "words" feature
    return [" ".join(email.words)] + common_features
