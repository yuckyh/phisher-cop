import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.lib.dataset import (
    hash_dir,
    hash_file,
    load_split,
    split_dir,
    unzip,
)


class TestDataset(unittest.TestCase):
    def test_hash_file(self):
        with TemporaryDirectory() as tmpdir:
            tmpfile_path = os.path.join(tmpdir, "test.txt")
            with open(tmpfile_path, "w") as f:
                f.write("Hello, world!")
            expected = (
                "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
            )
            actual = hash_file(tmpfile_path)
            self.assertEqual(actual, expected)

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
                "0abf1d129670de2fff96b5cd107c4ba326f07b50576f169e7087ddeddb5c75e4"
            )
            actual = hash_dir(tmpdir)
            self.assertEqual(actual, expected)

            os.rename(
                os.path.join(tmpdir, "dir1", "file3.txt"),
                os.path.join(tmpdir, "file3.txt"),
            )
            expected = (
                "bec2f2280253dc911759b097e719e280d3a83254cd19093bcbd5c5b9b5f8749c"
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

    def test_load_split(self):
        dir = Path(os.path.realpath(__file__)).parent
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(Exception):
                unzip(
                    os.path.join(dir, "data-split.zip"),
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    tmpdir,
                )

            unzip(
                os.path.join(dir, "data-split.zip"),
                "6cf7d664055b5a999f8552c57689062e9b8ce7b1ac8d600b37252bf36ba14920",
                tmpdir,
            )

            emails, labels = load_split(tmpdir)
            self.assertEqual(len(emails), 7)
            self.assertListEqual(
                labels.tolist(),
                [0, 0, 0, 0, 1, 1, 1],
            )
