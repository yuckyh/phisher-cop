"""Functions to sanitize and extract useful information from text documents."""

import urllib.parse
from email import message, message_from_file
from email.utils import parseaddr

from bs4 import BeautifulSoup, Tag
from html_sanitizer import Sanitizer

from .email_address import EmailAddress, parse_email_address

Email = message.Message


# TODO: reconsider this to be file upload specific
# e.g. different preprocessing logic for web input vs. file input
def email_from_file(path: str) -> Email:
    with open(path, "r", encoding="latin-1") as file:
        return message_from_file(file)


def email_from_input(
    sender: str,
    recipient: str,
    cc: list[str] | None,
    subject: str,
    payload: str,
):
    if not sender or not recipient or not subject or not payload:
        raise ValueError(
            "Sender, recipient, subject, and payload must not be empty or None."
        )
    if cc is None:
        cc = []
    email = Email()
    email["From"] = sender
    email["To"] = recipient
    email["Cc"] = ", ".join(cc)
    email["Subject"] = subject
    email.set_payload(payload)
    return email


def decode_payload(email: Email) -> str:
    assert not email.is_multipart()
    payload = email.get_payload(decode=True)
    return bytes(payload).decode(encoding="utf-8")


def payload_from_email(email: Email) -> str:
    if not email.is_multipart():
        return decode_payload(email)

    parts = [
        decode_payload(part)
        for part in email.walk()
        if part.get_content_type() in {"text/plain", "text/html"}
    ]
    return "\n".join(parts)


def sanitize_html(html: str) -> str:
    return Sanitizer().sanitize(html)


def sanitize_payload(email: Email) -> str:
    return sanitize_html(payload_from_email(email))


def urls_from_words(words: list[str]) -> set[urllib.parse.ParseResult]:
    parsed_urls = {urllib.parse.urlparse(word) for word in words}
    return {url for url in parsed_urls if url.netloc}


def anchor_urls_from_payload(payload: str) -> set[urllib.parse.ParseResult]:
    soup = BeautifulSoup(payload, "html.parser")
    return {
        urllib.parse.urlparse(str(a.get("href")))
        for a in soup.find_all("a", href=True)
        if isinstance(a, Tag)
    }


def urls_from_payload(payload: str) -> set[urllib.parse.ParseResult]:
    return urls_from_words(
        get_tokens(document_from_payload(payload))
    ) | anchor_urls_from_payload(payload)


def document_from_payload(payload: str) -> str:
    document = BeautifulSoup(payload, features="lxml").get_text(separator=" ")
    return document


def get_email_addresses(email: Email) -> list[EmailAddress]:
    try:
        return [
            parse_email_address(parseaddr(addr)[1])
            for field in ("From", "To", "Cc", "Bcc", "Reply-To")
            if email.get(field) is not None
            for addr in email[field].split(",")
        ]
    except AttributeError:
        pass


def get_tokens(document: str) -> list[str]:
    return [
        tokens.replace("\n", "")
        for tokens in document.split(" ")
        if tokens and tokens != "\n"
    ]


def get_words(tokens: list[str]) -> list[str]:
    urls = urls_from_words(tokens)
    url_strings = {urllib.parse.urlunparse(url) for url in urls}
    words = []
    for token in tokens:
        # Replace non-alphanumeric characters with spaces
        cleaned_token = "".join(" " if not char.isalnum() else char for char in token)
        for word in cleaned_token.split(" "):
            if word and token not in url_strings:
                words.append(word)
    return words
