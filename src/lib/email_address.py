import re
from dataclasses import dataclass
from email.utils import parseaddr
from urllib.parse import urlparse

from lib.domain import Domain, parse_domain


@dataclass()
class EmailAddress:
    """An email prepared for feature extraction."""

    username: str
    alias: str
    domain: Domain


ADDRESS_PATTERN = re.compile(r"(([^@+]*)\+)?([^@]+)@([^@]+)")


def parse_email_address(address: str | None) -> EmailAddress:
    """Parse an email address into its components."""
    if not isinstance(address, str):
        raise ValueError(f"Invalid email address: {address}")

    _, email_address = parseaddr(address)

    if not email_address:
        return EmailAddress(
            username="", alias="", domain=Domain(subdomain="", domain_name="", tld="")
        )

    match = ADDRESS_PATTERN.fullmatch(email_address)
    if not match:
        raise ValueError(f"Invalid email address: {email_address}")

    (_, alias, username, domain_str) = match.groups()
    alias = alias or ""

    url = urlparse("http://" + domain_str)
    domain = parse_domain(url)
    return EmailAddress(username=username, alias=alias, domain=domain)
