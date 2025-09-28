"""Functions to parse and extract useful information from email documents."""

import re
import urllib.parse
from email import message, message_from_bytes
from email.utils import getaddresses

from bs4 import BeautifulSoup, Tag

from lib.domain import Domain, Url, parse_domain
from lib.email_address import EmailAddress, parse_email_address

Email = message.Message


# TODO: reconsider this to be file upload specific
# e.g. different preprocessing logic for web input vs. file input
def email_from_file(path: str) -> Email:
    with open(path, "rb") as file:
        return message_from_bytes(file.read())


def email_from_input(
    sender: str,
    recipient: str,
    cc: list[str] | None,
    subject: str,
    payload: str,
) -> Email:
    if not sender or not recipient or not subject or not payload:
        raise ValueError(
            "Sender, recipient, subject, and payload must not be empty or None."
        )
    if cc is None:
        cc = []
    email = Email()
    email["From"] = sender
    email["To"] = recipient
    if cc:
        email["Cc"] = ", ".join(cc)
    email["Subject"] = subject
    email["Content-Type"] = "text/plain; charset=utf-8"
    email.set_payload(payload)
    return email


def decode_payload(email: Email) -> str:
    assert not email.is_multipart()
    # decode=True decodes transfer-encoding (e.g. base64, quoted-printable)
    payload = email.get_payload(decode=True)
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        # The payload is in some form of bytes, decode it using the email's charset
        content_charset = (
            (email.get_content_charset() or "utf-8")
            .removesuffix("_charset")
            .replace("chinesebig5", "big5")
            .replace("default", "utf-8")
        )
        return payload.decode(
            encoding=content_charset,
            errors="replace",
        )
    if isinstance(payload, str):
        return payload
    return str(payload)


def remove_payload_quotes(payload: str) -> str:
    """Removes common quoting prefixes from email payloads."""
    # This is a very naive implementation and may not cover all cases.
    # It only removes '>' characters at the start of lines.
    return "\n".join(line.lstrip("> ") for line in payload.splitlines())


def raw_payload(email: Email) -> str:
    if not email.is_multipart():
        return decode_payload(email)

    parts = [
        decode_payload(part)
        for part in email.walk()
        if part.get_content_type() in ("text/plain", "text/html")
    ]
    return "\n".join(parts)


def payload_dom(email: Email) -> BeautifulSoup:
    # payload is HTML or plain text, but plain text is a subset of HTML
    payload = remove_payload_quotes(raw_payload(email))
    return BeautifulSoup(payload, features="lxml")


def get_email_addresses(email: Email) -> list[EmailAddress]:
    addresses = []
    values = [
        value
        for field in ("From", "To", "Cc", "Bcc", "Reply-To")
        for value in email.get_all(field, [])
    ]
    for real_name, addr in getaddresses(values):
        try:
            addresses.append(parse_email_address(addr))
        except ValueError:
            # Skip invalid email addresses
            pass
    return addresses


def normalize_url(url: str) -> Url:
    """Normalizes a URL by lowercasing, unquoting, stripping trailing slashes
    and removing params, query, and fragment."""
    # Lowering must be done before unquoting because capital letters can be percent-encoded
    # Lowering must be done for correct string matching during feature extraction
    unquoted_url = urllib.parse.unquote(url.lower())
    parsed_url = urllib.parse.urlparse(unquoted_url)
    return Url(
        scheme=parsed_url.scheme,
        netloc=parsed_url.netloc,
        path=parsed_url.path.rstrip("/"),
        params="",
        query="",
        fragment="",
    )


def anchor_urls(dom: BeautifulSoup) -> set[Url]:
    """Returns a set of normalized URLs from the href attributes of anchor tags in the document."""
    return {
        url
        for anchor in dom.find_all("a", href=True)
        if isinstance(anchor, Tag)
        and isinstance(href := anchor.get("href"), str)
        # Only valid URLs have a network location
        and (url := normalize_url(href)).netloc
    }


def token_urls(
    raw_tokens: list[str],
) -> tuple[set[Url], list[str]]:
    """Iterates through `raw_tokens` and returns a set of normalized URLs and a list of non-URL tokens.
    The order of non-URL tokens is preserved."""
    urls = set()
    non_url_tokens = []
    for token in raw_tokens:
        url = normalize_url(token)
        if url.netloc:  # Only valid URLs have a network location
            urls.add(url)
        else:
            non_url_tokens.append(token)
    return urls, non_url_tokens


def raw_dom_tokens(dom: BeautifulSoup) -> list[str]:
    """Returns the whitespace-separated tokens of the document's text content."""
    return dom.get_text(separator=" ").split()


def domains_from_urls(urls: set[Url]) -> list[Domain]:
    return [parse_domain(url) for url in urls]


def tokenize_payload(email: Email) -> tuple[set[Url], list[str]]:
    """Returns a set of normalized URLs and a list of non-URL tokens from the email's payload.
    The order of non-URL tokens is preserved."""

    tokens: list[str] = []
    anchor_url_set = set()

    if email.get_content_type() == "text/plain":
        tokens = [token for token in remove_payload_quotes(raw_payload(email)).split()]
    else:
        dom_payload = payload_dom(email)
        tokens = raw_dom_tokens(dom_payload)
        anchor_url_set = anchor_urls(dom_payload)

    for token in tokens:
        if not isinstance(token, str):
            raise ValueError("Token is not a string: " + repr(token))
    urls, non_url_tokens = token_urls(tokens)
    urls |= anchor_url_set

    return urls, non_url_tokens


NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^a-z0-9]+",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


def words_from_tokens(tokens: list[str]) -> list[str]:
    """Returns all contiguous alphanumeric substrings from the tokens."""
    # Do NOT lowercase the words as some features are case-sensitive
    return [
        word
        for token in tokens
        for word in NON_ALPHANUMERIC_PATTERN.split(token)
        if word
    ]
