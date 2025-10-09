"""
Implementation of a Burkhard-Keller tree (BK-tree) for efficient string similarity searching.

This module provides:
1. BKTree data structure for fast approximate string matching
2. Levenshtein distance function for measuring string similarity
3. Efficient algorithms for finding strings within a specific edit distance

The BK-tree structure allows for much faster lookup of similar strings than
naive approaches, making it ideal for detecting typosquatted domains and
similar phishing techniques. The tree organizes strings in a way that allows
quick elimination of large portions of the search space.
"""

from dataclasses import dataclass

from typing_extensions import Callable, Iterable


@dataclass
class BKTreeNode:
    """
    Node in a BK-tree data structure.

    Each node stores a string label and its children, organized by their
    distance from the parent node. This structure enables efficient
    similarity searches by pruning branches that cannot contain matches.

    Attributes:
        label: The string stored at this node
        children: Dictionary mapping distances to child nodes
    """

    label: str
    children: dict[int, "BKTreeNode"]


class BKTree:
    """
    A Burkhard-Keller tree for efficiently finding close matches to strings.

    The BK-tree is a specialized data structure that enables fast approximate
    string matching by organizing strings according to their edit distance.
    This structure is particularly useful for:

    1. Finding all strings within a specific edit distance of a query string
    2. Detecting typosquatted domains in phishing detection
    3. Implementing fuzzy search functionality

    The tree's structure allows for efficient searching by pruning entire
    subtrees that cannot contain matches, significantly reducing the search space.
    """

    def __init__(
        self,
        distance_fn: Callable[[str, str], int],
        items: Iterable[str],
    ):
        """
        Create a BK-tree with the given `distance_fn` and initial `items`.
        `distance_fn(a, a)` must return `0` for all strings `a`.

        Time complexity: `O(n * log(n) * O(distance_fn))` on average, `O(n^2 * O(distance_fn))` in the worst case.
        Space complexity: `O(n)`

        Where `n = len(items)`.

        Example:
            >>> def simple_distance(a, b):
            ...     return abs(len(a) - len(b))
            >>> tree = BKTree(simple_distance, ["cat", "dog", "mouse"])
            >>> print(len(tree.items))
            3
            >>> print("cat" in tree.items)
            True
        """
        self.root: BKTreeNode | None = None
        self.items: set[str] = set()
        self.distance_fn = distance_fn
        for s in items:
            self.insert(s)

    def insert(self, item: str):
        """
        Insert `item` into the tree.

        Time complexity: `O(log(n) * O(distance_fn))` on average, `O(n * O(distance_fn))` in the worst case.
        Space complexity: `O(1)`

        Where `n` is the number of items in the tree.

        Example:
            >>> def simple_distance(a, b):
            ...     return abs(len(a) - len(b))
            >>> tree = BKTree(simple_distance, ["cat"])
            >>> tree.insert("dog")
            >>> print(len(tree.items))
            2
            >>> tree.insert("dog")  # Inserting duplicate has no effect
            >>> print(len(tree.items))
            2
            >>> tree.insert("mouse")
            >>> "mouse" in tree.items
            True
        """
        # Item is already in the tree
        if item in self.items:
            return

        self.items.add(item)
        if self.root is None:
            self.root = BKTreeNode(label=item, children={})
            return

        parent = self.root
        while True:
            distance = self.distance_fn(item, parent.label)
            # All descendants must have the same distance to the parent,
            child = parent.children.get(distance)
            # So item must be a descendant of child
            if child is not None:
                parent = child
                continue

            # No child with this distance, so we insert the item and finish
            parent.children[distance] = BKTreeNode(label=item, children={})
            break

    def contains_max_distance(self, item: str, max_distance: int) -> bool:
        """
        Check if the tree contains an item with distance at most `max_distance` from `item`.

        Time complexity: `O(log(n) * T_d)` on average, `O(n * T_d)` in the worst case.
        Space complexity: `O(log(n))` on average, `O(n)` in the worst case.

        Where `n` is the number of items in the tree and `T_d` is the time complexity of `distance_fn`.

        Example:
            >>> def simple_distance(a, b):
            ...     return abs(len(a) - len(b))
            >>> tree = BKTree(simple_distance, ["cat", "doggy", "mouse"])
            >>> tree.contains_max_distance("fish", 1)  # Only "cat" is within distance 1
            True
            >>> tree.contains_max_distance("a", 1)  # No strings of length 1 or 2
            False
        """
        assert max_distance >= 0
        if self.root is None:
            return False

        # Traverse the tree, starting from the root
        nodes_to_visit = [self.root]
        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop()
            distance = self.distance_fn(item, node.label)
            if distance <= max_distance:
                return True

            # Only visit children that could have distance <= max_distance
            lower_bound = distance - max_distance
            upper_bound = distance + max_distance
            nodes_to_visit.extend(
                child
                for d, child in node.children.items()
                if lower_bound <= d <= upper_bound
            )

        return False

    def __repr__(self) -> str:  # pragma: no cover
        return f"BKTree(root={self.root},distance_fn={self.distance_fn})"


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Time complexity: `O(len(s1) * len(s2))`
    Space complexity: `O(min(len(s1), len(s2)))`

    Example:
        >>> levenshtein_distance("cat", "bat")
        1
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("", "abc")
        3
    """
    # Make s2 the shorter string to use less memory
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    assert len(s1) >= len(s2)

    # Optimization for trivial case
    if len(s2) == 0:
        return len(s1)

    # First row of distance matrix is the distances from s1[:0] to s2's prefixes
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        # First element is the distance from s1[:i+1] to s2[:0]
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Levenshtein distance considers the operations insertion, deletion, and substitution
            insertion = previous_row[j + 1] + 1
            deletion = current_row[j] + 1
            substitution = previous_row[j] + (c1 != c2)
            # This is the Levenshtein distance for s1[:i+1] and s2[:j+1]
            current_row.append(min(insertion, deletion, substitution))
        # We computed the distances of s1[:i+1] and all prefixes of s2,
        # so we can move to the next row
        previous_row = current_row

    return previous_row[-1]
