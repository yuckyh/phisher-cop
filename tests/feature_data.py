import os
import unittest
from tempfile import TemporaryDirectory

from src.lib.feature_data import load_lines_as_set


class TestFeatureData(unittest.TestCase):
    def test_load_lines_as_set(self):
        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            with open(filepath, "w") as f:
                f.write(
                    "\n".join(
                        [
                            "Hello",
                            "world",
                            "",
                            "hello",
                            "",
                        ]
                    )
                )
            actual = load_lines_as_set(filepath, lower=True)
        expected = {"hello", "world"}
        self.assertSetEqual(actual, expected)
