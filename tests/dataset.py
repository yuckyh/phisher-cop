import os
import unittest
from tempfile import TemporaryDirectory

from src.dataset import hash_dir, split_dir


class TestDataset(unittest.TestCase):
    def test_hash_dir(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
                f.write("Hello, world!")
            with open(os.path.join(tmpdir, "file2.txt"), "w") as f:
                f.write("Another file.")
            os.makedirs(os.path.join(tmpdir, "dir1"))
            with open(os.path.join(tmpdir, "dir1", "file3.txt"), "w") as f:
                f.write("File in dir1.")

            expected = (
                "4d7963a2fb80742746c631ce4fb5d334971cbc5b202ba0c87cf92240b496951f"
            )
            actual = hash_dir(tmpdir)
            self.assertEqual(actual, expected)

            os.rename(
                os.path.join(tmpdir, "dir1", "file3.txt"),
                os.path.join(tmpdir, "file3.txt"),
            )
            expected = (
                "2383893af072f5aff2b67a12f5d4faa66aad6feb91e7f353450f586b3dd7c079"
            )
            actual = hash_dir(tmpdir)
            self.assertEqual(actual, expected)

    def test_split_dir(self):
        with TemporaryDirectory() as tmpdir:
            for i in range(10):
                with open(os.path.join(tmpdir, f"file{i}.txt"), "w") as f:
                    f.write(f"File {i}")

            splits = split_dir(tmpdir, [0.6, 0.2, 0.2])
            self.assertEqual(len(splits), 3)
            self.assertEqual(len(splits[0]), 6)
            self.assertEqual(len(splits[1]), 2)
            self.assertEqual(len(splits[2]), 2)

            all_files = set()
            for split in splits:
                all_files.update(split)
            self.assertEqual(len(all_files), 10)
