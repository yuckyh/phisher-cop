"""Module for parsing URLs into their components."""

import urllib.parse
from dataclasses import dataclass

from tldextract import extract

Url = urllib.parse.ParseResult
"""A parsed URL in lowercase."""


@dataclass()
class Domain:
    """A parsed domain name."""

    subdomain: str
    domain_name: str
    tld: str

    @property
    def host(self) -> str:
        """Return the host (domain + tld) of the domain."""
        return f"{self.domain_name}.{self.tld}" if self.tld else self.domain_name


def parse_domain(url: Url) -> Domain:
    """Return only the domain of a URL."""
    domain_parts = extract(url.netloc)
    return Domain(
        subdomain=domain_parts.subdomain,
        domain_name=domain_parts.domain,
        tld=domain_parts.registry_suffix,
    )
