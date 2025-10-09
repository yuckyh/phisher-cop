"""
Functions to parse and extract useful information from email documents.

This module provides utilities for:
1. Loading emails from files or constructing them from input
2. Preprocessing emails to extract features like URLs, text tokens, words
3. Parsing email addresses and domains
4. Extracting and normalizing URLs from email content
5. Tokenizing email content for feature extraction

The preprocessing pipeline is designed to handle both plain text
and HTML emails, with special handling for quoted content and
encodings commonly found in emails.

Libraries used:
- BeautifulSoup4: Used for parsing and navigating HTML content in emails
"""

import re
import urllib.parse
from dataclasses import dataclass
from email import message, message_from_bytes
from email.utils import getaddresses

from bs4 import BeautifulSoup, Tag

from .domain import Domain, Url, parse_domain
from .email_address import EmailAddress, parse_email_address

# Type alias for standard email.Message objects
Email = message.Message


@dataclass()
class PreprocessedEmail:
    """A preprocessed email with raw features extracted for analysis.

    This dataclass stores the results of email preprocessing, which are
    used as inputs for feature extraction. It contains all the raw data
    needed for phishing detection features.

    Attributes:
        urls: Set of normalized URLs found in the email
        tokens: List of raw tokens from the email text
        words: List of alphabetic words extracted from tokens
        sender: The parsed sender email address, or None if parsing failed
        addresses: List of all email addresses found in From/Cc fields
        domains: List of domain objects extracted from the URLs
    """

    urls: set[Url]
    tokens: list[str]
    words: list[str]
    sender: EmailAddress | None
    addresses: list[EmailAddress]
    domains: list[Domain]


def preprocess_email(email: Email, ignore_errors: bool = True) -> PreprocessedEmail:
    """Preprocess an email to extract features for phishing detection.

    This function extracts various components from an email that are useful
    for phishing detection, including URLs, text tokens, words, the sender,
    all email addresses, and domains.

    Args:
        email: The email message to preprocess
        ignore_errors: When True, errors parsing email addresses will be ignored
                      (sender will be None); when False, errors will be raised

    Returns:
        PreprocessedEmail: A dataclass containing all extracted features

    Raises:
        ValueError: If ignore_errors=False and email address parsing fails

    Example:
        >>> email_obj = email_from_file("data/test/ham/0001.txt")
        >>> processed = preprocess_email(email_obj)
        >>> print(len(processed.words) > 0)
        True
        >>> print(processed.sender is not None)
        True
    """
    urls, tokens = tokenize_payload(email)
    words = words_from_tokens(tokens)
    try:
        sender = parse_email_address(email["From"] or "")
    except ValueError as e:
        if not ignore_errors:
            raise e
        sender = None
    addresses = get_email_addresses(email, ignore_errors)
    domains = domains_from_urls(urls)
    return PreprocessedEmail(
        urls=urls,
        tokens=tokens,
        words=words,
        sender=sender,
        addresses=addresses,
        domains=domains,
    )


def email_from_file(path: str) -> Email:
    """Load an email message from a file.

    Args:
        path: Path to the email file

    Returns:
        Email: The parsed email message

    Example:
        >>> email = email_from_file("data/test/ham/0001.txt")
        >>> print(email["Subject"])
        'Re: New Sequences Window'
    """
    with open(path, "rb") as file:
        return message_from_bytes(file.read())


def email_from_input(
    sender: str,
    subject: str,
    payload: str,
    cc: str,
) -> Email:
    """
    Create an email message from user input components.

    This function is used primarily by the web interface to create
    an email object from user-supplied fields.

    Args:
        sender: The sender's email address
        subject: The email subject line
        payload: The email body content (HTML or plain text)
        cc: The CC field content (can be empty)

    Returns:
        Email: A constructed email message object

    Raises:
        ValueError: If sender, subject, or payload are empty

    Example:
        >>> email = email_from_input(
        ...     sender="john@example.com",
        ...     subject="Hello",
        ...     payload="<p>This is a test email.</p>",
        ...     cc="jane@example.com"
        ... )
        >>> print(email["From"])
        'john@example.com'
        >>> print(email["Subject"])
        'Hello'
    """
    if not sender or not subject or not payload:
        raise ValueError("Sender, subject, and payload must not be empty or None.")
    email = Email()
    email["From"] = sender
    email["Cc"] = cc
    email["Subject"] = subject
    # Assume the payload is HTML as plain text is a subset of HTML anyways
    email["Content-Type"] = "text/html; charset=utf-8"
    email.set_payload(payload)
    return email


def decode_payload(email: Email) -> str:
    """Decode the payload of a non-multipart email.

    This function handles decoding email content from various encodings and
    character sets, with special handling for common email encoding issues.

    Args:
        email: A non-multipart email message

    Returns:
        str: The decoded content as a string

    Raises:
        AssertionError: If the email is multipart or payload is not bytes
    """
    assert not email.is_multipart()
    # decode=True decodes transfer-encoding (e.g. base64, quoted-printable)
    payload = email.get_payload(decode=True)
    if payload is None:
        return ""

    assert isinstance(payload, bytes)
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


def get_email_addresses(email: Email, ignore_errors: bool) -> list[EmailAddress]:
    addresses = []
    values = [value for field in ("From", "Cc") for value in email.get_all(field, [])]
    for real_name, addr in getaddresses(values, strict=False):
        try:
            addresses.append(parse_email_address(addr))
        except ValueError as e:
            if not ignore_errors:
                raise e
    return addresses


def normalize_url(rawUrl: str) -> Url:
    """
    Normalize a URL for consistent comparison and analysis.

    This function standardizes URLs by:
    1. Converting to lowercase
    2. Unquoting percent-encoded characters
    3. Stripping trailing slashes from the path
    4. Removing params, query string, and fragment identifier

    These normalizations help with matching similar URLs and reducing false negatives
    in phishing detection.

    Args:
        rawUrl: The raw URL string to normalize

    Returns:
        Url: A normalized URL namedtuple

    Example:
        >>> url = normalize_url("HTTPS://Example.com/Path/?query=value#fragment")
        >>> print(url.scheme)
        'https'
        >>> print(url.netloc)
        'example.com'
        >>> print(url.path)
        '/path'
        >>> print(url.query)
        ''
    """
    # Lowering must be done before unquoting because capital letters can be percent-encoded
    # Lowering must be done for correct string matching during feature extraction
    unquoted_url = urllib.parse.unquote(rawUrl.lower())
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
    """
    Iterates through `raw_tokens` and returns a set of normalized URLs and a list of non-URL tokens.
    The order of non-URL tokens is preserved.
    """
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
    """
    Extract normalized URLs and text tokens from an email's payload.

    This function handles both plain text and HTML emails differently:
    - For plain text: splits the content by whitespace
    - For HTML: extracts text content and anchor href attributes

    In both cases, it identifies and separates URLs from regular text tokens.

    Args:
        email: The email message to tokenize

    Returns:
        tuple: A 2-tuple containing:
            - set[Url]: Set of normalized URLs found in the email
            - list[str]: List of non-URL tokens with order preserved
    """
    tokens: list[str] = []
    anchor_url_set = set()

    if email.get_content_type() == "text/plain":
        # For plain text emails, simply split by whitespace after removing quotes
        tokens = [token for token in remove_payload_quotes(raw_payload(email)).split()]
    else:
        # For HTML emails, use BeautifulSoup to extract text and URLs
        dom_payload = payload_dom(email)
        tokens = raw_dom_tokens(dom_payload)
        anchor_url_set = anchor_urls(dom_payload)

    # Extract URLs from tokens and separate them from non-URL tokens
    urls, non_url_tokens = token_urls(tokens)

    # Combine URLs found in tokens with those found in HTML anchors
    urls |= anchor_url_set

    return urls, non_url_tokens


NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^a-z0-9]+",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


def words_from_tokens(tokens: list[str]) -> list[str]:
    """
    Extract words (alphanumeric substrings) from a list of tokens.

    This function splits tokens at non-alphanumeric characters and
    returns all non-empty substrings. The case of words is preserved
    because some features (like capital_words_ratio) depend on case.

    Args:
        tokens: List of string tokens to process

    Returns:
        list[str]: All non-empty alphanumeric substrings found

    Example:
        >>> words_from_tokens(["Hello,", "world123!"])
        ['Hello', 'world123']
    """
    # Do NOT lowercase the words as some features are case-sensitive
    return [
        word
        for token in tokens
        for word in NON_ALPHANUMERIC_PATTERN.split(token)
        if word
    ]
