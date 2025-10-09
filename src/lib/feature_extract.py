"""
Feature extraction module for phishing detection.

Libraries used:
- typing_extensions: Enhanced type annotations for better code readability
"""

import re
from functools import lru_cache
from ipaddress import ip_address

from typing_extensions import Callable, Iterable, Iterator

from .bktree import BKTree, levenshtein_distance
from .domain import Domain, Url
from .email_address import EmailAddress
from .feature_data import load_suspicious_words, load_top_domains

# Define these outside the functions to avoid reloading the data on each call.
SAFE_DOMAINS = load_top_domains()
SAFE_DOMAIN_TREE = BKTree(levenshtein_distance, SAFE_DOMAINS)
SUSPICIOUS_WORDS = load_suspicious_words()


def count_whitelisted_addresses(
    emails: Iterable[EmailAddress],
    safe_domains: set[str],
) -> int:
    """
    Count how many email addresses come from known safe/trusted domains.

    This is an important feature for phishing detection because legitimate
    emails typically come from well-known domains, while phishing attempts
    often use unusual or deceptive domains.

    Args:
        emails: Collection of email addresses to check
        safe_domains: Set of domain names considered safe/trusted

    Returns:
        int: Count of email addresses from safe domains

    Time complexity: O(n) where n is the number of email addresses
    Space complexity: O(1)

    Example:
        >>> from .email_address import parse_email_address
        >>> emails = [
        ...     parse_email_address("user@google.com"),
        ...     parse_email_address("contact@phishing-site.com")
        ... ]
        >>> safe_domains = {"google.com", "microsoft.com", "apple.com"}
        >>> count_whitelisted_addresses(emails, safe_domains)
        1
    """
    return sum(1 for email in emails if email.domain.host in safe_domains)


def find_suspicious_words(
    words: Iterable[str],
    suspicious_words: set[str],
) -> Iterator[int]:
    """
    Scans the `words` for suspicious keywords and returns the index of each keyword found.
    For performance reasons, all words in `suspicious_words` must be lowercase.

    Time complexity: `O(n)` where `n` is the number of words.
    Space complexity: `O(1)`.
    """
    for i, word in enumerate(words):
        if word.lower() in suspicious_words:
            # Use a generator to reduce memory usage,
            # as this list is only used once to calculate a score.
            yield i


def suspicious_word_kernel(x: float) -> float:  # pragma: no cover
    """A kernel function that weights suspicious words based on their position.

    This function gives higher weight to suspicious words that appear earlier
    in the text, as phishing emails often front-load their suspicious content.

    Args:
        x: Normalized position in the text (0 = start, 1 = end)

    Returns:
        float: Weight for a word at this position (2.0 at start, 1.0 at end)

    Raises:
        AssertionError: If x is outside the range [0, 1]

    Example:
        >>> suspicious_word_kernel(0.0)  # Word at the beginning
        2.0
        >>> suspicious_word_kernel(0.5)  # Word in the middle
        1.5
        >>> suspicious_word_kernel(1.0)  # Word at the end
        1.0
    """
    # This kernel linearly interpolates between 2 at x=0 and 1 at x=1.
    assert 0 <= x <= 1
    return 2 - x


def score_suspicious_words(
    words: list[str],
    suspicious_words: set[str],
    kernel: Callable[[float], float] = suspicious_word_kernel,
) -> float:
    """Calculate a score for suspicious words found in the email.

    This function:
    1. Finds all suspicious words in the email
    2. Weights them based on their position using the kernel function
    3. Normalizes the score by the total number of words

    Words appearing earlier in the text get higher weights, as phishing
    emails often front-load their suspicious content.

    Args:
        words: List of words from the email
        suspicious_words: Set of known suspicious words (must be lowercase)
        kernel: Function that weights words by position (default: suspicious_word_kernel)

    Returns:
        float: A normalized suspiciousness score between 0.0 and 2.0
               (0.0 = no suspicious words, higher = more suspicious)
    """
    end = max(1, len(words) - 1)
    score = 0.0
    # Multiply y by the kernel and then integrate to get the score
    for index in find_suspicious_words(words, suspicious_words):
        x = index / end  # Normalize to [0, 1]
        # y is 1 when this is a suspicious word, and 0 otherwise.
        # Since anything multiplied by 0 is 0, we can skip non-suspicious words.
        y = kernel(x)
        # We need to multiply y by the step size 1 / len(words),
        # but it's more efficient to do it once at the end.
        score += y
    # Multiply by the step size 1 / len(words), but avoid division by zero.
    return score / max(1, len(words))


@lru_cache(maxsize=1000)
def is_typosquatted_domain(
    domain_host: str,
    safe_domain_tree: BKTree,
    edit_threshold: int,
) -> bool:
    """
    Check if a domain is a likely typosquatted version of a safe domain.

    Typosquatting is a technique used by phishers where they register domains
    that are very similar to legitimate domains but with small typos, like
    'goggle.com' instead of 'google.com'.

    This function uses Levenshtein distance to detect domains that are
    similar to safe domains but not identical.

    Example:
        >>> tree = BKTree(levenshtein_distance, ["google.com", "microsoft.com"])
        >>> is_typosquatted_domain("goggle.com", tree, 2)
        True
        >>> is_typosquatted_domain("completelydifferent.com", tree, 2)
        False

    Args:
        domain_host: The domain host name to check
        safe_domain_tree: BK-tree of safe domain names for efficient similarity search
        edit_threshold: Maximum edit distance to consider a match (typically 1-2)

    Returns:
        bool: True if the domain appears to be a typosquatted version of a safe domain
    """
    return (
        domain_host not in safe_domain_tree.items
        and safe_domain_tree.contains_max_distance(domain_host, edit_threshold)
    )


def count_typosquatted_domains(
    domains: Iterable[Domain],
    safe_domain_tree: BKTree,
    edit_threshold: int,
) -> int:
    """
    Count the number of likely typosquatted domains in an email.

    This is an important phishing detection feature as phishers often use
    domains that are visually similar to legitimate domains to trick users.

    Args:
        domains: Collection of domains to check
        safe_domain_tree: BK-tree of safe domain names for efficient similarity search
        edit_threshold: Maximum edit distance to consider a match (typically 1)

    Returns:
        int: Count of domains that appear to be typosquatted
    """
    return sum(
        1
        for domain in domains
        if is_typosquatted_domain(domain.host, safe_domain_tree, edit_threshold)
    )


def is_ip_address(url: Url) -> bool:
    """Return whether the URL's netloc is an IP address."""
    if url.hostname is None:
        return False
    try:
        _ = ip_address(url.hostname)
        return True
    except ValueError:
        return False


def count_ip_addresses(urls: Iterable[Url]) -> int:
    """Count the number of URLs that are IP addresses."""
    return sum(1 for url in urls if is_ip_address(url))


def email_domain_matches_url(
    email_address: EmailAddress | None,
    url_domains: list[Domain],
) -> bool:
    """Check if the email domain matches any of the given URL domains."""
    if email_address is None:
        # If we can't find the sender, there's something wrong with this email,
        # so we return False to mark it as suspicious.
        return False
    if len(url_domains) == 0:
        # If there are no URLs, we cannot match the sender's domain to any URL domain,
        # but we return True to mark it as safe as phishing emails usually contain URLs.
        return True
    for domain in url_domains:
        if email_address.domain.host == domain.host:
            return True
    return False


def capital_words_ratio(words: list[str]) -> float:
    """
    Calculate the ratio of all-uppercase words in the email.

    Phishing emails often use excessive capitalization to create urgency
    or draw attention to specific parts of the message.

    Args:
        words: List of words from the email

    Returns:
        float: Ratio of all-uppercase words to total words (0.0 to 1.0)
    """
    return sum(
        1  # This comment is to force the formatter to keep this on multiple lines
        for word in words
        if word.isalpha() and word.isupper()
    ) / max(1, len(words))


# Define this outside the function to avoid recompiling the regex on each call.
MONEY_PATTERN = re.compile(r"[$€£]\d+")


def money_tokens_ratio(tokens: list[str]) -> float:
    """
    Calculate the ratio of tokens that represent monetary amounts.

    Phishing emails often mention money to entice victims (e.g., prizes,
    refunds, or payments that need attention).

    Args:
        tokens: List of tokens from the email

    Returns:
        float: Ratio of money-related tokens to total tokens (0.0 to 1.0)
    """
    return sum(
        1  # This comment is to force the formatter to keep this on multiple lines
        for token in tokens
        if MONEY_PATTERN.match(token)
    ) / max(1, len(tokens))
