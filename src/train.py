"""Entry point for the model training script."""

import os
from pathlib import Path

from dataset import load_data

if __name__ == "__main__":
    project_root = Path(os.path.realpath(__file__)).parent.parent
    os.chdir(project_root)
    train, val, test = load_data()
    for split, name in zip((train, val, test), ("Train", "Validation", "Test")):
        print(f"{name} set : {len(split[0])} samples")

    raise Exception("TODO: Not implemented")
