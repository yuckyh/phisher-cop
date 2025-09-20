"""Functions to sanitize and extract useful information from text documents."""

import mailbox
from email.utils import parseaddr

from bs4 import BeautifulSoup, Tag
from html_sanitizer import Sanitizer

from email_address import EmailAddress, parse_email_address


# TODO: reconsider this to be file upload specific
# e.g. different preprocessing logic for web input vs. file input
def email_from_file(path: str) -> list[mailbox.mboxMessage]:
    return [
        email
        for email in mailbox.mbox(path)
        if email.is_multipart() or email["From"] is not None
    ]


def email_from_input(
    sender: str, recipient: str, cc: list[str], subject: str, payload: str
) -> mailbox.mboxMessage:
    email = mailbox.mboxMessage()
    email["From"] = sender
    email["To"] = recipient
    email["Cc"] = ", ".join(cc)
    email["Subject"] = subject
    email.set_payload(payload)
    return email


def payload_from_email(email: mailbox.mboxMessage) -> str:
    if not email.is_multipart():
        return str(email.get_payload())

    parts = []
    for part in email.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "\n".join(parts)


def sanitize_html(html: str) -> str:
    return Sanitizer().sanitize(html)


def sanitize_payload(email: mailbox.mboxMessage) -> str:
    return sanitize_html(payload_from_email(email))


def urls_from_payload(payload: str) -> set[str]:
    soup = BeautifulSoup(payload, "html.parser")
    urls = {
        str(a.get("href")) for a in soup.find_all("a", href=True) if isinstance(a, Tag)
    }
    return urls


def document_from_payload(payload: str) -> str:
    document = BeautifulSoup(payload, features="lxml").get_text(separator=" ")
    return document


def get_email_addresses(email: mailbox.mboxMessage) -> list[EmailAddress]:
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


def get_words(document: str) -> list[str]:
    return [word for word in document.split(" ") if word]
