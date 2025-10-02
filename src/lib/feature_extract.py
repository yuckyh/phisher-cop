import re
from ipaddress import ip_address

from typing_extensions import Iterable, Iterator

from lib.bktree import BKTree, levenshtein_distance
from lib.domain import Domain, Url
from lib.email import PreprocessedEmail
from lib.email_address import EmailAddress
from lib.feature_data import load_suspicious_words, load_top_domains

# Define these outside the functions to avoid reloading the data on each call.
SAFE_DOMAINS = load_top_domains()
SAFE_DOMAIN_TREE = BKTree(levenshtein_distance, SAFE_DOMAINS)
SUSPICIOUS_WORDS = load_suspicious_words()


def count_whitelisted_addresses(emails: Iterable[EmailAddress]) -> int:
    """
    Count how many email addresses are from a whitelisted domain.

    Time complexity: `O(n)` where `n` is the number of email addresses.
    Space complexity: `O(1)`.
    """
    return sum(1 for email in emails if email.domain.host in SAFE_DOMAINS)


def find_suspicious_words(words: Iterable[str]) -> Iterator[int]:
    """
    Scans the `words` for suspicious keywords and returns the index of each keyword found.

    Time complexity: `O(n)` where `n` is the number of words.
    Space complexity: `O(1)`.
    """
    for i, word in enumerate(words):
        if word.lower() in SUSPICIOUS_WORDS:
            # Use a generator to reduce memory usage,
            # as this list is only used once to calculate a score.
            yield i


def suspicious_word_kernel(x: float) -> float:
    """
    A kernel function that gives higher weight to words appearing earlier in the text.
    `x` is a normalized position in the text in the range [0, 1].
    """
    # This kernel linearly interpolates between 2 at x=0 and 1 at x=1.
    assert 0 <= x <= 1
    return 2 - x


def score_suspicious_words(words: list[str]) -> float:
    """
    Score the suspicious words in the given list of words.
    Higher scores are given to suspicious words that appear earlier in the list.
    This score is normalized so that the length of the list does not affect it.
    """
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
    # Multiply by the step size 1 / len(words), but avoid division by zero.
    return score / max(1, len(words))


def count_typosquatted_domains(
    domains: Iterable[Domain],
    edit_threshold: int,
) -> int:
    """Count the number of domains that are likely typosquatted versions of popular domains."""
    return sum(
        1
        for domain in domains
        if domain.host not in SAFE_DOMAINS
        and SAFE_DOMAIN_TREE.contains_max_distance(domain.host, edit_threshold)
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


# def sender_domain_matches_url(email: Email, url_domains: list[Domain]) -> bool:
def email_domain_matches_url(
    email_address: EmailAddress, url_domains: list[Domain]
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
    """Returns the ratio of words in all caps to those that are not."""
    return sum(1 for word in words if all(c.isupper() for c in word)) / max(
        1, len(words)
    )


# Define this outside the function to avoid recompiling the regex on each call.
MONEY_PATTERN = re.compile(r"[$€£]\d+")


def money_tokens_ratio(tokens: list[str]) -> float:
    """Returns the ratio of tokens that represent money amounts to those that do not."""
    return sum(1 for token in tokens if MONEY_PATTERN.match(token)) / max(
        1, len(tokens)
    )


def extract_features(email: PreprocessedEmail) -> list[float]:
    # Order of these features are very IMPORTANT.
    return [
        float(count_whitelisted_addresses(email.addresses)),
        score_suspicious_words(email.words),
        float(count_typosquatted_domains(email.domains, edit_threshold=1)),
        float(count_ip_addresses(email.urls)),
        1.0 if email_domain_matches_url(email.sender, email.domains) else 0.0,
        capital_words_ratio(email.words),
        money_tokens_ratio(email.tokens),
    ]
