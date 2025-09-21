import joblib
from sklearn.gaussian_process import GaussianProcessClassifier

Model = GaussianProcessClassifier


def save_model(model: Model, path: str) -> None:
    joblib.dump(model, path, compress=("xz", 7))


def load_model(path: str) -> Model:
    return joblib.load(path)
