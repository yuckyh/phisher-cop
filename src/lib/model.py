"""
Core model functionality for the phishing detection system.

This module provides the core machine learning infrastructure for phishing detection:
- ModelType enum to represent different classifier types
- PhisherCop class for end-to-end email phishing detection
- Functions for model creation, preprocessing, and feature extraction

Example:
    >>> from lib.model import ModelType, PhisherCop
    >>> from lib.email import email_from_input
    >>>
    >>> # Load a trained phishing detection model
    >>> model = PhisherCop.load("models/svm.joblib")
    >>>
    >>> # Create an email from user input
    >>> email = email_from_input(
    ...     sender="suspicious@example.com",
    ...     subject="URGENT: Verify your account now!",
    ...     payload="Click here to verify: http://192.168.1.1/login",
    ...     cc=""
    ... )
    >>>
    >>> # Get phishing probability score
    >>> score = model.score_email(email)
    >>> print(f"Phishing probability: {score:.1%}")
    Phishing probability: 87.5%
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

MODELS_PATH = os.path.join(
    PROJECT_ROOT, "models"
)  # Path to the directory containing trained model files
Model = RandomForestClassifier | SVC  # Type alias for supported model types


class ModelType(Enum):
    """
    Enum defining the available machine learning model types for phishing detection.

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
        """
        Determine if the model type requires TF-IDF text features.

        Returns:
            bool: True if the model type uses TF-IDF features, False otherwise

        Example:
            >>> ModelType.RANDOM_FOREST.uses_tfidf
            True
            >>> ModelType.SVM.uses_tfidf
            False
            >>> # This can be used to decide whether to include text features
            >>> for model_type in ModelType:
            ...     print(f"{model_type.name}: {'Uses' if model_type.uses_tfidf else 'Does not use'} TF-IDF")
            RANDOM_FOREST: Uses TF-IDF
            SVM: Does not use TF-IDF
        """
        match self:
            case ModelType.RANDOM_FOREST:
                return True
            case ModelType.SVM:
                return False

    @property
    def default_path(self) -> str:  # pragma: no cover
        """
        Get the default file path for saving/loading this model type.

        Returns:
            str: The absolute path where models of this type are stored by default

        Example:
            >>> import os
            >>> # Get default paths for different model types
            >>> svm_path = ModelType.SVM.default_path
            >>> rf_path = ModelType.RANDOM_FOREST.default_path
            >>>
            >>> # Paths include the model type name
            >>> os.path.basename(svm_path)
            'svm.joblib'
            >>> os.path.basename(rf_path)
            'random_forest.joblib'
            >>>
            >>> # Use in loading/saving operations
            >>> model = PhisherCop.load(ModelType.SVM.default_path)
            >>> model.save(ModelType.SVM.default_path)
        """
        return os.path.join(MODELS_PATH, f"{self.value}.joblib")


class PhisherCop:
    """
    End-to-end inference for e-mail phishing detection.

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
        """
        Initialize the PhisherCop model with a preprocessor pipeline and trained model.

        Args:
            pipeline: A scikit-learn Pipeline that transforms raw features into model inputs
            model: A trained classifier model (RandomForestClassifier or SVC)

        Raises:
            ValueError: If the model type is not supported

        Example:
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.preprocessing import StandardScaler
            >>> from sklearn.svm import SVC
            >>>
            >>> # Create pipeline and model
            >>> pipeline = Pipeline([('scaler', StandardScaler())])
            >>> model = SVC(probability=True)
            >>>
            >>> # Initialize PhisherCop instance
            >>> phisher_cop = PhisherCop(pipeline, model)
            >>> phisher_cop.model_type == ModelType.SVM
            True
            >>>
            >>> # Trying with unsupported model type
            >>> try:
            ...     PhisherCop(pipeline, "not a model")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Unsupported model type
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
        """
        Save the PhisherCop model to disk.

        Args:
            path: File path where the model will be saved

        Returns:
            None

        Example:
            >>> import os
            >>> from sklearn.pipeline import Pipeline
            >>> from sklearn.svm import SVC
            >>>
            >>> # Create and save a model
            >>> pipeline = Pipeline([('noop', 'passthrough')])
            >>> model = SVC(probability=True)
            >>> phisher_cop = PhisherCop(pipeline, model)
            >>>
            >>> # Save to a temp directory
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model_path = os.path.join(tmpdir, "test_model.joblib")
            ...     phisher_cop.save(model_path)
            ...     print(f"Model saved: {os.path.exists(model_path)}")
            Model saved: True
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path, compress=("zlib", 3))  # type: ignore

    @staticmethod
    def load(path: str) -> "PhisherCop":
        """
        Load a saved PhisherCop model from disk.

        Args:
            path: File path to the saved model

        Returns:
            PhisherCop: The loaded model

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the loaded object is not a PhisherCop instance

        Example:
            >>> # Load an existing model
            >>> try:
            ...     model = PhisherCop.load("models/svm.joblib")
            ...     print(f"Model type: {model.model_type.name}")
            ... except FileNotFoundError:
            ...     print("Model file not found (expected in testing environment)")
            ...
            >>> # Handling non-existent file
            >>> try:
            ...     PhisherCop.load("non_existent_file.joblib")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
            Error: Model file not found: non_existent_file.joblib
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)
        if not isinstance(model, PhisherCop):
            raise ValueError("Loaded object is not a PhisherCop instance")
        return model

    def score_email(self, email: Email) -> float:
        """
        Predict the probability that an email is a phishing attempt.

        Args:
            email: An Email object to be scored

        Returns:
            float: Phishing score between 0.0 and 1.0, where:
                  - 1.0 means definitely spam/phishing
                  - 0.0 means definitely ham/legitimate

        Example:
            >>> from lib.email import email_from_input
            >>>
            >>> # Load a model (or create a sample one for testing)
            >>> try:
            ...     model = PhisherCop.load("models/svm.joblib")
            ... except FileNotFoundError:
            ...     # For demonstration, create a simple model
            ...     from sklearn.pipeline import Pipeline
            ...     from sklearn.preprocessing import StandardScaler
            ...     from sklearn.svm import SVC
            ...     pipeline = Pipeline([('scaler', StandardScaler())])
            ...     clf = SVC(probability=True)
            ...     model = PhisherCop(pipeline, clf)
            ...
            >>> # Create test emails with different risk profiles
            >>> legitimate_email = email_from_input(
            ...     "friend@gmail.com",
            ...     "Lunch tomorrow",
            ...     "Hi! Are we still on for lunch tomorrow? Let me know. Thanks!",
            ...     ""
            ... )
            >>>
            >>> suspicious_email = email_from_input(
            ...     "banking@secure-verify-login.com",
            ...     "URGENT: Verify Your Account Now",
            ...     "Dear customer, your account has been locked. Click here: http://182.168.0.1/login",
            ...     ""
            ... )
            >>>
            >>> # Compare scores (actual values will vary)
            >>> legitimate_score = model.score_email(legitimate_email)
            >>> suspicious_score = model.score_email(suspicious_email)
            >>> print(f"Legitimate email is {'suspicious' if legitimate_score > 0.5 else 'safe'}")
            >>> print(f"Suspicious email is {'suspicious' if suspicious_score > 0.5 else 'safe'}")
        """
        preprocessed_email = preprocess_email(email, ignore_errors=False)
        features = extract_features(self.model_type, preprocessed_email)
        features = self.pipeline.transform([features])
        return self.model.predict_proba(features)[0, Label.SPAM.value].item()  # type: ignore


def create_preprocessor(model_type: ModelType) -> Pipeline:
    """
    Create an untrained preprocessor pipeline for the specified model type.

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
        Model: An untrained classifier instance

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.svm import SVC
        >>> rf_model = create_model(ModelType.RANDOM_FOREST, 42)
        >>> isinstance(rf_model, RandomForestClassifier)
        True
        >>> svm_model = create_model(ModelType.SVM, 42)
        >>> isinstance(svm_model, SVC)
        True
        >>> svm_model.probability  # SVM configured to output probabilities
        True
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

    The types of features extracted depend on the model type.

    Args:
        model_type: The type of model for which features are being extracted
        email: A preprocessed email containing various extracted components

    Returns:
        list: A feature vector containing numerical features (and text for TF-IDF models)

    Example:
        >>> from lib.email import PreprocessedEmail
        >>> # Create a sample preprocessed email
        >>> email = PreprocessedEmail(
        ...     words=["hello", "money", "urgent", "account"],
        ...     tokens=["hello", "money", "$100", "urgent", "account"],
        ...     urls=["http://example.com", "http://192.168.1.1"],
        ...     domains=["example.com"],
        ...     addresses=["user@example.com", "admin@gmail.com"],
        ...     sender="user@malicious.com"
        ... )
        >>> # Extract features for SVM model (no TF-IDF)
        >>> svm_features = extract_features(ModelType.SVM, email)
        >>> len(svm_features)  # Seven numerical features
        7
        >>> isinstance(svm_features[0], float)  # All features are floats
        True
        >>> # Extract features for Random Forest model (with TF-IDF)
        >>> rf_features = extract_features(ModelType.RANDOM_FOREST, email)
        >>> len(rf_features)  # Eight features (text + seven numerical)
        8
        >>> isinstance(rf_features[0], str)  # First feature is text
        True
        >>> rf_features[0]  # Space-joined words
        'hello money urgent account'
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
