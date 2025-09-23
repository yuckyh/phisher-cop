from enum import Enum
from ipaddress import ip_address

from typing_extensions import Iterable

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
