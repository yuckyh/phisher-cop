import re
from enum import Enum

from typing_extensions import Iterable

from .document import Email
from .domain import Url
from .email_address import parse_email_address


class HostType(Enum):
    IP = 0
    DOMAIN = 1
    UNKNOWN = 2


IP_ADDRESS_PATTERN = re.compile(
    r"(^(?:\d{1,3}\.){3}\d{1,3}$)"  # IPv4
    "|"
    r"(^\[?[0-9a-fA-F:]+\]?$)"  # IPv6
)


def host_type(host: str) -> HostType:
    """Return the type of the given host string (IP address or domain)."""
    return HostType.DOMAIN if IP_ADDRESS_PATTERN.match(host) is None else HostType.IP


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
