import unittest

from src.lib import parallelize

# These weird comments are to stop ruff from removing the "unused" imports
from .bktree import TestBKTree, TestLevenshteinDistance  # noqa: F401
from .dataset import TestDataset  # noqa: F401
from .domain import TestDomain  # noqa: F401
from .email import TestDocument  # noqa: F401
from .feature_extract import TestFeatureExtract  # noqa: F401


class TestParallelize(unittest.TestCase):
    def test_parallelize(self):
        addTwo = lambda x: x + 2
        input = [1, 2, 3, 4, 5]
        expected = list(map(addTwo, input))
        actual = parallelize(addTwo, input)
        self.assertEqual(expected, actual)
