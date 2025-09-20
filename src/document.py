"""Functions to sanitize and extract useful information from text documents."""

from domain import Domain
from email_address import EmailAddress, parse_email_address

import mailbox
from email.utils import parseaddr

from html_sanitizer import Sanitizer
from bs4 import BeautifulSoup, Tag


# TODO: reconsider this to be file upload specific
# e.g. different preprocessing logic for web input vs. file input
def email_from_file(path: str) -> list[mailbox.mboxMessage]:
    return [email for email in mailbox.mbox(path) if email.is_multipart() or email['From'] is not None]


def email_from_input(sender: str, recipient: str, cc: list[str], subject: str, payload: str) -> mailbox.mboxMessage:
    email = mailbox.mboxMessage()
    email['From'] = sender
    email['To'] = recipient
    email['Cc'] = ', '.join(cc)
    email['Subject'] = subject
    email.set_payload(payload)
    return email


def payload_from_email(email: mailbox.mboxMessage) -> str:
    if email.is_multipart():
        parts = []
        for part in email.walk():
            if part.get_content_type() == "text/plain":
                parts.append(part.get_payload())
        return "\n".join(parts)

    return str(email.get_payload())


def sanitize_html(html: str) -> str:
    return Sanitizer().sanitize(html)


def sanitize_payload(email: mailbox.mboxMessage) -> str:
    return sanitize_html(payload_from_email(email))


def urls_from_payload(payload: str) -> set[str]:
    soup = BeautifulSoup(payload, 'html.parser')
    urls = {str(a.get('href')) for a in soup.find_all('a', href=True) if isinstance(a, Tag)}
    return urls


def document_from_payload(payload: str) -> str:
    document = BeautifulSoup(payload, features='lxml').get_text(separator=' ')
    return document


def get_email_addresses(email: mailbox.mboxMessage) -> list[EmailAddress]:
    try:
        return [parse_email_address(parseaddr(addr)[1]) for field in ('From', 'To', 'Cc', 'Bcc', 'Reply-To') for addr in email[field].split(',')]
    except AttributeError:
        pass
    except ValueError:
        pass
    return []


def get_words(document: str) -> list[str]:
    return [word for word in document.split(' ') if word]


def get_urls(words: list[str]) -> list[Domain]:
    raise NotImplementedError


if __name__ == "__main__":
    emails = email_from_file("data/train/spam/0029.txt")
    email = emails[0]
    print(len(emails))
    print(email['Subject'])
    print(get_email_addresses(email))

    sanitized_payload = sanitize_payload(email)
    document = document_from_payload(sanitized_payload)

    print(urls_from_payload(sanitized_payload))

    words = get_words(document)
    print(words)