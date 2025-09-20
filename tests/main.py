import unittest

# These weird comments are to stop ruff from removing the "unused" imports
from tests.dataset import TestDataset  # noqa: F401
from tests.document import TestDocument  # noqa: F401
from tests.feature_extract import TestFeatureExtract  # noqa: F401

if __name__ == "__main__":
    unittest.main()
