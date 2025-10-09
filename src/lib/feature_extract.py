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
    Scan a list of words for suspicious keywords and return the indices of matches.

    This function efficiently identifies suspicious words that might indicate phishing content
    by comparing each word against a known set of suspicious terms. It returns the positions
    where suspicious words are found.

    Time complexity: `O(n)` where `n` is the number of words.
    Space complexity: `O(1)`.

    Args:
        words: Iterable of words to check against suspicious words list
        suspicious_words: Set of suspicious words to look for (must be lowercase)

    Returns:
        Iterator[int]: Generator yielding indices of suspicious words in the input list


    Example:
        >>> words = ["Hello", "update", "your", "password", "now"]
        >>> suspicious = {"password", "update", "urgent", "verify"}
        >>> list(find_suspicious_words(words, suspicious))
        [1, 3]
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

    Args:
        domain_host: The domain host name to check
        safe_domain_tree: BK-tree of safe domain names for efficient similarity search
        edit_threshold: Maximum edit distance to consider a match (typically 1-2)

    Returns:
        bool: True if the domain appears to be a typosquatted version of a safe domain

    Example:
        >>> tree = BKTree(levenshtein_distance, ["google.com", "microsoft.com"])
        >>> is_typosquatted_domain("goggle.com", tree, 2)
        True
        >>> is_typosquatted_domain("completelydifferent.com", tree, 2)
        False
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
    For example, using "goggle.com" instead of "google.com" or
    "paypa1.com" instead of "paypal.com".

    Args:
        domains: Collection of domains to check
        safe_domain_tree: BK-tree of safe domain names for efficient similarity search
        edit_threshold: Maximum edit distance to consider a match (typically 1)

    Returns:
        int: Count of domains that appear to be typosquatted
\
    Example:
        >>> from .domain import Domain
        >>> from .bktree import BKTree, levenshtein_distance
        >>> safe_domains = ["google.com", "amazon.com", "microsoft.com"]
        >>> tree = BKTree(levenshtein_distance, safe_domains)
        >>> domains = [
        ...     Domain("www", "goggle", "com"),
        ...     Domain("", "amaz0n", "com"),
        ...     Domain("", "totally-different", "org")
        ... ]
        >>> count_typosquatted_domains(domains, tree, 1)
        2
    """
    return sum(
        1
        for domain in domains
        if is_typosquatted_domain(domain.host, safe_domain_tree, edit_threshold)
    )


def is_ip_address(url: Url) -> bool:
    """
    Determine whether a URL uses an IP address instead of a domain name.

    URLs that use IP addresses directly (like http://192.168.1.1/) instead of domain names
    are often associated with phishing attempts, as they circumvent DNS lookups and can
    hide the true identity of the server.

    Args:
        url: A parsed URL object to check

    Returns:
        bool: True if the URL uses an IP address, False otherwise

    Example:
        >>> from urllib.parse import urlparse
        >>> is_ip_address(urlparse("http://192.168.1.1/login"))
        True
        >>> is_ip_address(urlparse("https://example.com/path"))
        False
        >>> is_ip_address(urlparse("http://2001:db8::1/path"))  # IPv6
        True
    """
    if url.hostname is None:
        return False
    try:
        _ = ip_address(url.hostname)
        return True
    except ValueError:
        return False


def count_ip_addresses(urls: Iterable[Url]) -> int:
    """
    Count the number of URLs that use IP addresses instead of domain names.

    URLs with IP addresses are often a red flag in phishing detection because legitimate
    organizations typically use domain names. Phishers may use IP addresses to avoid
    domain registration or to disguise the true location of their servers.

    Args:
        urls: Collection of URL objects to check

    Returns:
        int: Number of URLs that use IP addresses

    Example:
        >>> from urllib.parse import urlparse
        >>> urls = [
        ...     urlparse("http://192.168.1.1/login"),
        ...     urlparse("https://example.com/path"),
        ...     urlparse("http://10.0.0.1/secure")
        ... ]
        >>> count_ip_addresses(urls)
        2
    """
    return sum(1 for url in urls if is_ip_address(url))


def email_domain_matches_url(
    email_address: EmailAddress | None,
    url_domains: list[Domain],
) -> bool:
    """
    Check if the sender's email domain matches any domains in the URLs of the email.

    This is an important phishing indicator because legitimate emails typically contain
    links to the sender's own domain (e.g., emails from amazon.com link to amazon.com),
    while phishing emails often have mismatched domains (e.g., emails claiming to be
    from amazon.com link to malicious-site.com).

    Args:
        email_address: The parsed sender email address, or None if parsing failed
        url_domains: List of domains extracted from URLs in the email

    Returns:
        bool: True if the sender's domain matches any URL domain or there are no URLs,
              False if there's no match or the sender couldn't be parsed

    Example:
        >>> from .domain import Domain
        >>> from .email_address import parse_email_address
        >>> sender = parse_email_address("service@amazon.com")
        >>> domains = [
        ...     Domain(subdomain="www", domain_name="amazon", tld="com"),
        ...     Domain(subdomain="", domain_name="phishing", tld="com")
        ... ]
        >>> email_domain_matches_url(sender, domains)
        True
        >>> domains = [Domain(subdomain="", domain_name="phishing", tld="com")]
        >>> email_domain_matches_url(sender, domains)
        False
    """
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
    or draw attention to specific parts of the message. This function helps
    identify emails that use this tactic by measuring the proportion of
    words that are entirely uppercase.

    Args:
        words: List of words from the email

    Returns:
        float: Ratio of all-uppercase words to total words (0.0 to 1.0)

    Example:
        >>> capital_words_ratio(["Hello", "URGENT", "please", "UPDATE", "now"])
        0.4
        >>> capital_words_ratio(["this", "is", "normal", "text"])
        0.0
        >>> capital_words_ratio(["ALL", "CAPS", "TEXT", "HERE"])
        1.0
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
    refunds, or payments that need attention). This function detects tokens
    that represent currency amounts using common currency symbols ($, €, £)
    followed by numbers.

    Args:
        tokens: List of tokens from the email

    Returns:
        float: Ratio of money-related tokens to total tokens (0.0 to 1.0)

    Example:
        >>> money_tokens_ratio(["You", "won", "$1000", "in", "prizes"])
        0.2
        >>> money_tokens_ratio(["Your", "bill", "is", "$50", "please", "pay", "€20", "now"])
        0.25
        >>> money_tokens_ratio(["No", "money", "mentioned", "here"])
        0.0
    """
    return sum(
        1  # This comment is to force the formatter to keep this on multiple lines
        for token in tokens
        if MONEY_PATTERN.match(token)
    ) / max(1, len(tokens))
