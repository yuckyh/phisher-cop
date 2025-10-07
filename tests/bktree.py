import unittest

from src.lib.bktree import BKTree, levenshtein_distance


class TestLevenshteinDistance(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("flaw", "lawn"), 2)
        self.assertEqual(levenshtein_distance("intention", "execution"), 5)
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("a", ""), 1)
        self.assertEqual(levenshtein_distance("", "a"), 1)
        self.assertEqual(levenshtein_distance("abc", "abc"), 0)
        self.assertEqual(levenshtein_distance("abc", "abx"), 1)
        self.assertEqual(levenshtein_distance("abc", "axc"), 1)
        self.assertEqual(levenshtein_distance("abc", "xbc"), 1)
        self.assertEqual(levenshtein_distance("abc", "xyz"), 3)


class TestBKTree(unittest.TestCase):
    def test_insert(self):
        tree = BKTree(levenshtein_distance, [])
        tree.insert("hello")
        self.assertIsNotNone(tree.root)
        assert tree.root is not None  # Fix type checker warnings
        self.assertEqual(tree.root.label, "hello")

        tree.insert("hallo")
        self.assertIn(1, tree.root.children)
        self.assertEqual(tree.root.children[1].label, "hallo")
        self.assertEqual(len(tree.items), 2)

        tree.insert("hallo")
        self.assertEqual(len(tree.items), 2)

        tree.insert("hell")
        self.assertIn(1, tree.root.children)
        self.assertIn(2, tree.root.children[1].children)
        self.assertEqual(tree.root.children[1].children[2].label, "hell")

        tree.insert("allo")
        self.assertIn(2, tree.root.children)
        self.assertEqual(tree.root.children[2].label, "allo")

    def test_contains_max_distance(self):
        tree = BKTree(levenshtein_distance, ["hello", "hallo", "hell", "allo", "help"])

        self.assertTrue(tree.contains_max_distance("hello", 0))  # "hello"
        self.assertTrue(tree.contains_max_distance("hallo", 1))  # "hallo"
        self.assertTrue(tree.contains_max_distance("hell", 1))  # "hell"
        self.assertFalse(tree.contains_max_distance("world", 3))
        # "world" -> "horld" -> "herld" -> "helld" -> "hello"
        self.assertTrue(tree.contains_max_distance("world", 4))
        self.assertFalse(tree.contains_max_distance("test", 2))
        # "test" -> "hest" -> "helt" -> "help"
        self.assertTrue(tree.contains_max_distance("test", 3))

        tree = BKTree(levenshtein_distance, [])
        self.assertFalse(tree.contains_max_distance("anything", 100))
