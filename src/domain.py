"""Module for parsing URLs into their components."""

from dataclasses import dataclass
from urllib.parse import urlparse

from tldextract import extract


@dataclass()
class Domain:
    """A parsed domain name."""

    subdomain: str
    domain_name: str
    tld: str


def parse_domain(url: str) -> Domain:
    """Parse a URL string and return only the domain."""
    url_parts = urlparse(url)
    domain_parts = extract(url_parts.netloc)
    return Domain(
        subdomain=domain_parts.subdomain,
        domain_name=domain_parts.domain,
        tld=domain_parts.registry_suffix,
    )
