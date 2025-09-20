"""Entry point for the model training script."""

import numpy as np
from numpy.typing import NDArray

from model import Model, save_model
from phisher_cop import MODEL_PATH


def dummy_data(
    rng: np.random.Generator, rows: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    features = rng.standard_normal((rows, 10))
    labels = features.sum(axis=1) > 0
    labels = labels.astype(np.int64)
    return features, labels


if __name__ == "__main__":
    # TODO: use real data instead of dummy data
    # os.chdir(project_root)
    # train, val, test = load_data()
    # for split, name in zip((train, val, test), ("Train", "Validation", "Test")):
    #     print(f"{name} set: {len(split[0])} samples")

    rng = np.random.default_rng(1974827191289312837)
    train, val, test = dummy_data(rng, 1000), dummy_data(rng, 200), dummy_data(rng, 200)
    ml = Model()
    ml.fit(*train)
    save_model(ml, MODEL_PATH)
    print(f"Train accuracy: {ml.score(*train):.3f}")
    print(f"Validation accuracy: {ml.score(*val):.3f}")
    print(f"Test accuracy: {ml.score(*test):.3f}")
