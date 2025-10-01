import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

MODEL_SEED = 69420
FORCE_RETRAIN = True

Model = LinearSVC


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    joblib.dump(pipeline, path, compress=("xz", 7))  # type: ignore


def try_load(path: str) -> Model | Pipeline | None:
    try:
        if FORCE_RETRAIN:
            raise FileNotFoundError("Force retrain enabled")
        return joblib.load(path)
    except FileNotFoundError as e:
        print(f"File not found at {path}, creating a new one. ({e})")
        return None


def load_pipeline(path: str) -> Pipeline:
    """Load or create the ML pipeline."""
    pipeline = try_load(path)
    if pipeline and isinstance(pipeline, Pipeline):
        return pipeline

    text_features = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("scaler", StandardScaler(with_mean=False)),  # Standardize features
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("text", text_features, 0),
        ],
        remainder=StandardScaler(),
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            # ("scaler", StandardScaler(with_mean=False)),  # Standardize features
            # ("classifier", LinearSVC(random_state=MODEL_SEED, tol=1e-4, max_iter=5000, C=0.01)),
        ]
    )
    return pipeline


def save_model(model: Model, path: str) -> None:
    joblib.dump(model, path, compress=("xz", 7))  # type: ignore


def load_model(path: str) -> Model:
    model = try_load(path)
    if model and isinstance(model, Model):
        return model

    return LinearSVC(random_state=MODEL_SEED, tol=1e-4, max_iter=5000, C=0.01)
