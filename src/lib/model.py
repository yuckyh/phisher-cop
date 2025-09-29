import joblib
from sklearn.base import TransformerMixin

# from sklearn.calibration import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_SEED = 69420
FORCE_RETRAIN = True
GRID = {
    "transformer__tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
    "transformer__tfidf__max_features": [100, 300, 500, 700, 1000],
    "transformer__tfidf__lowercase": [True, False],
    "transformer__tfidf__norm": ["l1", "l2", None],
    "transformer__tfidf__stop_words": [None, "english"],
    "model__random_state": [MODEL_SEED],
    "model__tol": [1e-2, 1e-3, 1e-4, 1e-5],
    "model__C": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
    "model__max_iter": [2000, 5000, 10000],
}

Model = HalvingGridSearchCV


class DenseTransformer(TransformerMixin):
    """Convert sparse matrix to dense. Used in a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()


def build_pipeline() -> Pipeline:
    words_pipeline = Pipeline(
        [("tfidf", TfidfVectorizer()), ("to_dense", DenseTransformer())]
    )

    column_transformers = ColumnTransformer(
        [
            ("words", words_pipeline, 0),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("transformer", column_transformers),
            ("scaler", StandardScaler()),
            # ("model", LinearSVC()),
            ("model", SGDClassifier())
        ], verbose = True
    )
    return pipeline


def build_cv(pipeline: Pipeline) -> HalvingGridSearchCV:
    cv = HalvingGridSearchCV(
        pipeline, GRID, n_jobs=-1, verbose=1, cv=5, random_state=MODEL_SEED
    )
    return cv


def save_model(model: Model, path: str) -> None:
    joblib.dump(model, path, compress=("xz", 7))  # type: ignore


def load_model(path: str) -> Model:
    try:
        if FORCE_RETRAIN:
            raise FileNotFoundError("Forcing retrain of model")
        return joblib.load(path)
    except FileNotFoundError:
        cv = build_cv(build_pipeline())

        return cv
