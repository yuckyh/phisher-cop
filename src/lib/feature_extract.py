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

    # TODO: get a longer list
    SUSPICIOUS_KEYWORDS = {
        "urgent",
        "hurry",
        "verify",
        "account",
        "action required",
        "password",
        "login",
        "security alert",
        "suspicious activity",
        "limited time",
        "low price",
        "immediate",
        "invoice",
        "payment",
        "click here",
        "link",
        "update",
        "confirm",
        "free",
    }

    for i, word in enumerate(words):
        if word.lower() in SUSPICIOUS_KEYWORDS:
            # Use a generator to reduce memory usage,
            # as this list is only used once to calculate a score.
            yield i


def score_suspicious_words(indices: Iterable[int]) -> float:
    raise NotImplementedError("TODO: implement scoring function")
