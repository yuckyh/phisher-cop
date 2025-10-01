import joblib
from sklearn.svm import LinearSVC

MODEL_SEED = 69420
FORCE_RETRAIN = True

Model = LinearSVC


def save_model(model: Model, path: str) -> None:
    joblib.dump(model, path, compress=("xz", 7))  # type: ignore


def load_model(path: str) -> Model:
    try:
        if FORCE_RETRAIN:
            raise FileNotFoundError("Force retrain enabled")
        return joblib.load(path)
    except FileNotFoundError as e:
        print(f"Model not found at {path}, creating a new one. ({e})")
        ml = LinearSVC(random_state=MODEL_SEED, tol=1e-4, max_iter=5000, C=0.01)

        return ml
