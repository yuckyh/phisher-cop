import unittest

from src.lib import parallelize

# Use redundant import names to make the linter happy
from .bktree import TestBKTree as TestBKTree
from .bktree import TestLevenshteinDistance as TestLevenshteinDistance
from .dataset import TestDataset as TestDataset
from .domain import TestDomain as TestDomain
from .email import TestEmail as TestEmail
from .feature_data import TestFeatureData as TestFeatureData
from .feature_extract import TestFeatureExtract as TestFeatureExtract


def addTwo(x: int) -> int:
    return x + 2


class TestParallelize(unittest.TestCase):
    def test_parallelize(self):
        input = [1, 2, 3, 4, 5]
        expected = list(map(addTwo, input))
        actual = parallelize(addTwo, input)
        self.assertEqual(expected, actual)
