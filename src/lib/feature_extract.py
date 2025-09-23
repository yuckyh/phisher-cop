import re

from .document import email_addresses, payload_dom, tokenize_dom

# from .email_address import parse_email_address
from .domain import parse_domain

# Regex for IPv4 and IPv6
IPV4_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
IPV6_PATTERN = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")


def is_ip_address(host: str) -> bool:
    """Check if the given host string is an IP address (IPv4 or IPv6)."""
    return bool(IPV4_PATTERN.match(host) or IPV6_PATTERN.match(host))


def sender_domain_type(email) -> str:
    """
    Check if the sender's domain is a normal domain or an IP.
    Returns "ip" or "domain".
    """
    addrs = email_addresses(email)
    if not addrs:
        return "unknown"
    sender = addrs[0]  # First address is usually the sender
    host = sender.domain.domain_name if sender.domain.domain_name else sender.domain.tld
    return "ip" if is_ip_address(host) else "domain"


def body_url_types(email) -> list[tuple[str, str]]:
    """
    Extract all URLs from the email body and check whether each is a domain or IP.
    Returns list of tuples (url, "ip"/"domain").
    """
    dom = payload_dom(email)
    urls, _ = tokenize_dom(dom)
    results = []
    for url in urls:
        parsed = parse_domain(url.geturl())  # reuse your domain parser
        host = parsed.domain_name if parsed.domain_name else parsed.tld
        label = "ip" if is_ip_address(host) else "domain"
        results.append((url.geturl(), label))
    return results
