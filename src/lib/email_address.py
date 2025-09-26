from dataclasses import dataclass
from urllib.parse import urlparse

from lib.domain import Domain, parse_domain


@dataclass()
class EmailAddress:
    """An email prepared for feature extraction."""

    username: str
    alias: str
    domain: Domain


def parse_email_address(address: str) -> EmailAddress:
    """Parse an email address into its components."""
    if "@" not in address:
        raise ValueError(f"Invalid email address: {address}")

    username, domain_str = address.split("@", 1)
    if "+" in username:
        username, alias = username.split("+", 1)
    else:
        alias = ""

    url = urlparse("http://" + domain_str)
    domain = parse_domain(url)

    return EmailAddress(username=username, alias=alias, domain=domain)
