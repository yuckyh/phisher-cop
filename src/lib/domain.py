"""
Module for parsing and analyzing URLs and domain names.

This module provides utilities for:
1. Parsing URLs into their component parts
2. Extracting and analyzing domain information from URLs
3. Representing domain names with their subdomain, main domain, and TLD components

The module helps identify and process domains for phishing detection by providing
structured access to domain components that can be analyzed for suspicious patterns.

Libraries used:
- tldextract: Extracts top-level domain, domain name and subdomain parts from URLs
"""

import urllib.parse
from dataclasses import dataclass

from tldextract import extract

Url = urllib.parse.ParseResult
"""A parsed URL in lowercase."""


@dataclass()
class Domain:
    """A parsed domain name with its component parts.

    This dataclass represents a domain name split into its structural components:
    subdomain, main domain name, and top-level domain (TLD). This structure
    facilitates analysis for phishing detection by allowing separate evaluation
    of each component.

    Attributes:
        subdomain: The subdomain portion (e.g., 'www' in 'www.example.com')
        domain_name: The main domain name (e.g., 'example' in 'www.example.com')
        tld: The top-level domain (e.g., 'com' in 'www.example.com')
    """

    subdomain: str
    domain_name: str
    tld: str

    @property
    def host(self) -> str:
        """Return the host (domain + tld) of the domain.

        Returns:
            str: The combined domain_name and tld (e.g., 'example.com')

        Example:
            >>> domain = Domain(subdomain='www', domain_name='example', tld='com')
            >>> print(domain.host)
            'example.com'
            >>> domain_no_tld = Domain(subdomain='', domain_name='localhost', tld='')
            >>> print(domain_no_tld.host)
            'localhost'
        """
        return f"{self.domain_name}.{self.tld}" if self.tld else self.domain_name


def parse_domain(url: Url) -> Domain:
    """Extract and parse the domain components from a URL.

    This function takes a parsed URL and extracts its domain information,
    splitting it into subdomain, main domain name, and TLD components.

    Args:
        url: A parsed URL object (from urllib.parse)

    Returns:
        Domain: A Domain object containing the parsed components

    Example:
        >>> from urllib.parse import urlparse
        >>> url = urlparse('https://www.example.com/path')
        >>> domain = parse_domain(url)
        >>> print(domain.subdomain)
        'www'
        >>> print(domain.domain_name)
        'example'
        >>> print(domain.tld)
        'com'
    """
    domain_parts = extract(url.netloc)
    return Domain(
        subdomain=domain_parts.subdomain,
        domain_name=domain_parts.domain,
        tld=domain_parts.registry_suffix,
    )
