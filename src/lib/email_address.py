from dataclasses import dataclass

from .domain import Domain, parse_domain


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

    domain = parse_domain("http://" + domain_str)

    return EmailAddress(username=username, alias=alias, domain=domain)
