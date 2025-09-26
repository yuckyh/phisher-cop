from enum import Enum
from ipaddress import ip_address

from typing_extensions import Iterable, Iterator

from .document import Email
from .domain import Url
from .email_address import parse_email_address


class HostType(Enum):
    IP = 0
    DOMAIN = 1
    UNKNOWN = 2


def host_type(host: str) -> HostType:
    """Return the type of the given host string (IP address or domain)."""
    try:
        ip_address(host)
        return HostType.IP
    except ValueError:
        return HostType.DOMAIN


def sender_domain_type(email: Email) -> HostType:
    """Check if the sender's domain is a normal domain or an IP."""
    sender = parse_email_address(email["Sender"])
    if not sender:
        return HostType.UNKNOWN
    return host_type(sender.domain.host)


def url_types(urls: Iterable[Url]) -> list[HostType]:
    """Check whether each URL is a domain or IP."""
    return [
        host_type(url.hostname) if url.hostname else HostType.UNKNOWN for url in urls
    ]


def find_suspicious_words(words: Iterable[str]) -> Iterator[int]:
    """
    Scans the `words` for suspicious keywords and returns the index of each keyword found.

    Args:
        words (Iterable[str]): The words to scan.

    Yields:
        Iterator[int]: The index of each suspicious keyword found.
    """
    SUSPICIOUS_KEYWORDS = {
        "free",
        "please",
        "send",
        "address",
        "information",
        "order",
        "email",
        "report",
        "make",
        "business",
        "money",
        "receive",
        "internet",
        "name",
        "click",
        "over",
        "home",
        "site",
    }

    for i, word in enumerate(words):
        if word.lower() in SUSPICIOUS_KEYWORDS:
            # Use a generator to reduce memory usage,
            # as this list is only used once to calculate a score.
            yield i


def _suspicious_word_kernel(x: float) -> float:
    """A kernel function that gives higher weight to words appearing earlier in the text.

    Args:
        x: A normalized position in the text in the range [0, 1].
    """
    # This kernel linearly interpolates between 2 at x=0 and 1 at x=1.
    assert 0 <= x <= 1
    return 2 - x


def score_suspicious_words(words: list[str]) -> float:
    end = len(words) - 1
    score = 0.0
    # Multiply by the kernel and then integrate to get the score
    for index in find_suspicious_words(words):
        x = index / end  # Normalize to [0, 1]
        # x is 1 when this is a suspicious word, and 0 otherwise.
        # Since anything multiplied by 0 is 0, we can skip non-suspicious words.
        y = _suspicious_word_kernel(x)
        # We need to multiply y by the step size 1 / len(words),
        # but it's more efficient to do it once at the end.
        score += y
    return score / len(words)
