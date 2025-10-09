"""
Core model functionality for the phishing detection system.

Libraries used:
- sklearn (scikit-learn): Machine learning library providing:
  - RandomForestClassifier: Ensemble method for classification
  - SVC: Support Vector Machine classifier
  - Pipeline: For creating machine learning workflows
  - ColumnTransformer: For applying transformations to specific columns
  - TfidfVectorizer: For text feature extraction
  - StandardScaler: For feature normalization
- joblib: For model serialization and persistence
"""

import os
from enum import Enum

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import PROJECT_ROOT
from .dataset import Label
from .email import Email, PreprocessedEmail, preprocess_email
from .feature_extract import (
    SAFE_DOMAIN_TREE,
    SAFE_DOMAINS,
    SUSPICIOUS_WORDS,
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
    """Enum defining the available machine learning model types for phishing detection.

    Each model type has different characteristics:
    - RANDOM_FOREST: Uses TF-IDF features and is generally more interpretable
    - SVM: Uses only numerical features and may handle certain datasets better

    The enum provides helper properties for determining feature requirements
    and default file paths for each model type.
    """
    RANDOM_FOREST = "random_forest"
    SVM = "svm"

    @property
    def uses_tfidf(self) -> bool:  # pragma: no cover
        """Determine if the model type requires TF-IDF text features.

        Returns:
            bool: True if the model type uses TF-IDF features, False otherwise
        """
        match self:
            case ModelType.RANDOM_FOREST:
                return True
            case ModelType.SVM:
                return False

    @property
    def default_path(self) -> str:  # pragma: no cover
        """Get the default file path for saving/loading this model type.

        Returns:
            str: The absolute path where models of this type are stored by default
        """
        return os.path.join(MODELS_PATH, f"{self.value}.joblib")


class PhisherCop:
    """End-to-end inference for e-mail phishing detection.

    This class provides a complete solution for phishing detection by combining:
    1. A preprocessing pipeline that transforms raw emails into feature vectors
    2. A trained machine learning model that classifies emails as phishing or legitimate

    The class handles loading/saving models and provides an easy-to-use interface
    for scoring emails to determine their phishing probability.

    Example:
        >>> from .email import email_from_file
        >>> model = PhisherCop.load("models/svm.joblib")
        >>> email = email_from_file("data/test/spam/0001.txt")
        >>> score = model.score_email(email)
        >>> print(score > 0.5)  # Is it likely to be phishing?
        True
    """

    def __init__(self, pipeline: Pipeline, model: Model) -> None:
        """Initialize the PhisherCop model with a preprocessor pipeline and trained model.

        Args:
            pipeline: A scikit-learn Pipeline that transforms raw features into model inputs
            model: A trained classifier model (RandomForestClassifier or SVC)

        Raises:
            ValueError: If the model type is not supported
        """
        self.pipeline = pipeline
        self.model = model

        # Infer model type from the model instance
        match model:
            case RandomForestClassifier():
                self.model_type = ModelType.RANDOM_FOREST
            case SVC():
                self.model_type = ModelType.SVM
            case _:
                raise ValueError("Unsupported model type")

    def save(self, path: str) -> None:
        """Save the PhisherCop model to disk.

        Args:
            path: File path where the model will be saved

        Returns:
            None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path, compress=("zlib", 3))  # type: ignore

    @staticmethod
    def load(path: str) -> "PhisherCop":
        """Load a saved PhisherCop model from disk.

        Args:
            path: File path to the saved model

        Returns:
            PhisherCop: The loaded model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the loaded object is not a PhisherCop instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)
        if not isinstance(model, PhisherCop):
            raise ValueError("Loaded object is not a PhisherCop instance")
        return model

    def score_email(self, email: Email) -> float:
        """Calculate the probability that an email is a phishing attempt.

        Args:
            email: An Email object to be scored

        Returns:
            float: Phishing score between 0.0 and 1.0, where:
                  - 1.0 means definitely spam/phishing
                  - 0.0 means definitely ham/legitimate
        """
        preprocessed_email = preprocess_email(email, ignore_errors=False)
        features = extract_features(self.model_type, preprocessed_email)
        features = self.pipeline.transform([features])
        return self.model.predict_proba(features)[0, Label.SPAM.value].item()  # type: ignore


def create_preprocessor(model_type: ModelType) -> Pipeline:
    """Create an untrained preprocessor pipeline for the specified model type.

    This function builds a scikit-learn Pipeline that handles:
    - For TF-IDF models: Text vectorization and numerical feature scaling
    - For non-TF-IDF models: Only numerical feature scaling

    Args:
        model_type: The type of model to create a preprocessor for

    Returns:
        Pipeline: An untrained scikit-learn preprocessing pipeline

    Example:
        >>> pipeline = create_preprocessor(ModelType.RANDOM_FOREST)
        >>> print(isinstance(pipeline, Pipeline))
        True
        >>> pipeline = create_preprocessor(ModelType.SVM)
        >>> print(isinstance(pipeline, Pipeline))
        True
    """
    if model_type.uses_tfidf:
        # For models using TF-IDF (like RandomForest), create a text processing pipeline
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
        # For models not using TF-IDF (like SVM), only scale numerical features
        pipeline = Pipeline(
            [
                ("preprocessor", StandardScaler()),
            ]
        )
    return pipeline


def create_model(model_type: ModelType, seed: int) -> Model:
    """
    Create an untrained model of the specified type.

    Args:
        model_type: The type of model to create
        seed: Random seed for reproducible results

    Returns:
        Model: An untrained classifier instance (RandomForestClassifier or SVC)
    """
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
    """
    Extract features from a preprocessed email for model training or inference.

    This function creates a feature vector containing:
    1. Count of email addresses from safe domains
    2. Score of suspicious words present in the email
    3. Count of typosquatted domains in the email
    4. Count of IP addresses in URLs
    5. Whether the sender's domain matches any URL domain
    6. Ratio of capitalized words
    7. Ratio of money-related tokens
    8. For TF-IDF models only: A space-joined string of all words

    Args:
        model_type: The type of model for which features are being extracted
        email: A preprocessed email containing various extracted components

    Returns:
        list: A feature vector containing numerical features (and text for TF-IDF models)
    """
    common_features: list[float | str] = [
        float(count_whitelisted_addresses(email.addresses, SAFE_DOMAINS)),
        score_suspicious_words(email.words, SUSPICIOUS_WORDS),
        float(
            count_typosquatted_domains(
                email.domains,
                SAFE_DOMAIN_TREE,
                edit_threshold=1,
            )
        ),
        float(count_ip_addresses(email.urls)),
        1.0 if email_domain_matches_url(email.sender, email.domains) else 0.0,
        capital_words_ratio(email.words),
        money_tokens_ratio(email.tokens),
    ]

    if not model_type.uses_tfidf:
        return common_features
    # TF-IDF requires extra "words" feature
    return [" ".join(email.words)] + common_features
