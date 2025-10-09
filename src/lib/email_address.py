"""
Module for parsing and analyzing email addresses.

This module provides utilities for:
1. Parsing email addresses into their component parts
2. Extracting username, alias, and domain components
3. Representing email addresses in a structured format for feature extraction

The module helps process email addresses and provides structured access to
components that can be analyzed.
"""

import re
from dataclasses import dataclass
from email.utils import parseaddr
from urllib.parse import urlparse

from .domain import Domain, parse_domain


@dataclass()
class EmailAddress:
    """
    A parsed email address prepared for feature extraction.

    This dataclass represents an email address split into its structural components:
    username, alias (if present), and domain. This structure facilitates analysis
    for phishing detection by allowing separate evaluation of each component.

    Attributes:
        username: The main part of the email address before the @ symbol
                 (excluding any alias part)
        alias: The optional plus-addressing part (e.g., 'alias' in 'user+alias@example.com')
              or an empty string if not present
        domain: The Domain object containing the parsed domain components
    """

    username: str
    alias: str
    domain: Domain


ADDRESS_PATTERN = re.compile(r"(([^@+]*)\+)?([^@]+)@([^@]+)")


def parse_email_address(address: str) -> EmailAddress:
    """
    Parse an email address into its component parts.

    This function takes an email address string and breaks it down into:
    1. Username (main part before the @ symbol)
    2. Alias (optional plus-addressing part)
    3. Domain (parsed into subdomain, domain name, and TLD)

    The function handles plus-addressing (user+alias@domain.com) and
    extracts a structured representation of the email address for analysis.

    Args:
        address: The email address string to parse

    Returns:
        EmailAddress: A structured representation of the email address

    Raises:
        ValueError: If the email address is empty or has an invalid format

    Example:
        >>> result = parse_email_address("john.doe+newsletter@example.com")
        >>> print(result.username)
        'john.doe'
        >>> print(result.alias)
        'newsletter'
        >>> print(result.domain.host)
        'example.com'
    """
    _, email_address = parseaddr(address)

    if not email_address:
        raise ValueError(f"Invalid email address: {address}")

    match = ADDRESS_PATTERN.fullmatch(email_address)
    if not match:
        raise ValueError(f"Invalid email address: {email_address}")

    (_, alias, username, domain_str) = match.groups()
    alias = alias or ""

    url = urlparse("http://" + domain_str)
    domain = parse_domain(url)
    return EmailAddress(username=username, alias=alias, domain=domain)
