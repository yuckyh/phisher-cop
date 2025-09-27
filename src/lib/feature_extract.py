from ipaddress import ip_address

from typing_extensions import Iterable, Iterator

from lib.bktree import BKTree, levenshtein_distance
from lib.document import Email
from lib.domain import Domain, Url
from lib.email_address import EmailAddress, parse_email_address
from lib.feature_data import load_suspicious_words, load_top_domains

SAFE_DOMAINS = load_top_domains()
SAFE_DOMAIN_TREE = BKTree(levenshtein_distance, SAFE_DOMAINS)
SUSPICIOUS_WORDS = load_suspicious_words()


def count_whitelisted_addresses(emails: Iterable[EmailAddress]) -> int:
    """Count how many email addresses are from a whitelisted domain.

    Time complexity: `O(n)` where `n` is the number of email addresses.
    Space complexity: `O(1)`.
    """
    return sum(1 for email in emails if email.domain.host in SAFE_DOMAINS)


def find_suspicious_words(words: Iterable[str]) -> Iterator[int]:
    """
    Scans the `words` for suspicious keywords and returns the index of each keyword found.

    Time complexity: `O(n)` where `n` is the number of words.
    Space complexity: `O(1)`.

    Args:
        words (Iterable[str]): The words to scan.

    Yields:
        Iterator[int]: The index of each suspicious keyword found.
    """
    for i, word in enumerate(words):
        if word.lower() in SUSPICIOUS_WORDS:
            # Use a generator to reduce memory usage,
            # as this list is only used once to calculate a score.
            yield i


def suspicious_word_kernel(x: float) -> float:
    """A kernel function that gives higher weight to words appearing earlier in the text.

    Args:
        x: A normalized position in the text in the range [0, 1].
    """
    # This kernel linearly interpolates between 2 at x=0 and 1 at x=1.
    assert 0 <= x <= 1
    return 2 - x


def score_suspicious_words(words: list[str]) -> float:
    """Score the suspicious words in the given list of words.
    Higher scores are given to suspicious words that appear earlier in the list."""
    end = max(1, len(words) - 1)
    score = 0.0
    # Multiply y by the kernel and then integrate to get the score
    for index in find_suspicious_words(words):
        x = index / end  # Normalize to [0, 1]
        # y is 1 when this is a suspicious word, and 0 otherwise.
        # Since anything multiplied by 0 is 0, we can skip non-suspicious words.
        y = suspicious_word_kernel(x)
        # We need to multiply y by the step size 1 / len(words),
        # but it's more efficient to do it once at the end.
        score += y
    return score / max(1, len(words))


def count_typosquatted_domains(
    domains: Iterable[Domain],
    edit_treshold: int,
) -> int:
    """Count the number of domains that are likely typosquatted versions of popular domains."""
    return sum(
        1
        for domain in domains
        if domain.host not in SAFE_DOMAINS
        and SAFE_DOMAIN_TREE.contains_max_distance(domain.host, edit_treshold)
    )


def is_ip_address(url: Url) -> bool:
    """Return whether the URL's netloc is an IP address."""
    try:
        _ = ip_address(url.netloc)
        return True
    except ValueError:
        return False


def count_ip_addresses(urls: Iterable[Url]) -> int:
    """Count the number of URLs that are IP addresses."""
    return sum(1 for url in urls if is_ip_address(url))


def sender_domain_matches_url(email: Email, url_domains: Iterable[Domain]) -> bool:
    """Check if the sender's domain matches any of the given URL domains."""
    sender = parse_email_address(email["Sender"])
    for domain in url_domains:
        if sender.domain.host == domain.host:
            return True
    return False


def count_capital_words(words: list[str]) -> int:
    """Count the number of words in all caps."""
    return sum(1 for word in words if all(c.isupper() for c in word))
