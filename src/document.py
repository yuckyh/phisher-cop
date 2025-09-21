"""Functions to sanitize and extract useful information from text documents."""

import urllib.parse
from email import message, message_from_file
from email.utils import parseaddr

from bs4 import BeautifulSoup, Tag
from html_sanitizer import Sanitizer

from email_address import EmailAddress, parse_email_address


# TODO: reconsider this to be file upload specific
# e.g. different preprocessing logic for web input vs. file input
def email_from_file(path: str) -> message.Message:
    with open(path, "r", encoding="latin-1") as file:
        return message_from_file(file)


def email_from_input(
    sender: str,
    recipient: str,
    cc: list[str],
    subject: str,
    payload: str,
) -> message.Message:
    email = message.Message()
    email["From"] = sender
    email["To"] = recipient
    email["Cc"] = ", ".join(cc)
    email["Subject"] = subject
    email.set_payload(payload)
    return email


def payload_from_email(email: message.Message) -> str:
    if not email.is_multipart():
        return str(email.get_payload())

    parts = [
        str(part.get_payload())
        for part in email.walk()
        if part.get_content_type() == "text/plain"
    ]
    return "\n".join(parts)


def sanitize_html(html: str) -> str:
    return Sanitizer().sanitize(html)


def sanitize_payload(email: message.Message) -> str:
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
    return urls_from_words(get_tokens(document_from_payload(payload))) | anchor_urls_from_payload(payload)


def document_from_payload(payload: str) -> str:
    document = BeautifulSoup(payload, features="lxml").get_text(separator=" ")
    return document


def get_email_addresses(email: message.Message) -> list[EmailAddress]:
    try:
        return [
            parse_email_address(parseaddr(addr)[1])
            for field in ("From", "To", "Cc", "Bcc", "Reply-To")
            for addr in email[field].split(",")
        ]
    except AttributeError:
        pass
    except ValueError:
        pass
    return []


def get_tokens(document: str) -> list[str]:
    return [tokens.replace('\n', '') for tokens in document.split(' ') if tokens and tokens != '\n']


def get_words(tokens: list[str]) -> list[str]:
    urls = urls_from_words(tokens)
    words = [word for token in tokens for word in ''.join(' ' if not char.isalnum() else char for char in token).split(' ') if word and token not in {urllib.parse.urlunparse(url) for url in urls}]

    return words


if __name__ == "__main__":
    email = email_from_file("data/test/ham/0087.txt")
    print("From:", email['From'])
    print("To:", email['To'])
    print("Cc:", email['Cc'])
    print("Subject:", email['Subject'])
    print()
    print("Payload:")
    print(payload_from_email(email))
    print()
    print("Sanitized Payload:")
    print(sanitize_payload(email))
    print()
    print("Document:")
    document = document_from_payload(payload_from_email(email))
    print(document)
    print()
    print("Words:")
    words = get_tokens(document)
    print(words)
    print()
    print("Tokens:")
    tokens = get_words(words)
    print(tokens)
    print()
    print("URLs:")
    for url in urls_from_payload(payload_from_email(email)):
        print(url)
    print()
    print("Email Addresses:")
    for address in get_email_addresses(email):
        print(address)
