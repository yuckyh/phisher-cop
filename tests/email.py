import os
import quopri
import unittest
from base64 import b64encode
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from src.lib.email import (
    email_from_file,
    payload_dom,
    raw_payload,
    tokenize_payload,
    words_from_tokens,
)


class TestDocument(unittest.TestCase):
    def test_email_from_file(self):
        expected_from = "Mail Delivery Subsystem <postmaster@example.com>"
        expected_content_type = "text/plain"
        expected_charset = "latin-1"
        expected_subject = "Undelivered Mail Returned to Sender"
        expected_payload = (
            "This is the mail system at host example.com. It's attached below.\r\n"
            "\r\n"
            "--- Original message ---\r\n"
        )
        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_mail.txt")
            with open(filepath, "w") as f:
                f.write(
                    f"From: {expected_from}\r\n"
                    f"To: recipient@example.com\r\n"
                    f"Content-Type: {expected_content_type}; charset={expected_charset}\r\n"
                    f"Subject: {expected_subject}\r\n"
                    "\r\n"
                    f"{expected_payload}"
                )
            email = email_from_file(filepath)
        self.assertEqual(email["From"], expected_from)
        self.assertEqual(email.get_content_type(), expected_content_type)
        self.assertEqual(email.get_content_charset(), expected_charset)
        self.assertEqual(email["Subject"], expected_subject)
        self.assertEqual(email.get_payload(), expected_payload)

    def test_raw_payload(self):
        expected_payload_plain = (
            "This is the plaintext version\r\n"
            '"quoted-printable" breaks up lines like this that are at least 76 characters long\r\n'
            "你好，世界！\r\n"
        )
        expected_payload_html = (
            "<!DOCTYPE html>\r\n"
            "<html>\r\n"
            "  <body>\r\n"
            "    <h1>This is the HTML version</h1>\r\n"
            "    <p>你好，世界！</p>\r\n"
            "  </body>\r\n"
            "</html>\r\n"
        )
        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_mail.txt")
            with open(filepath, "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r\n"
                    "To: recipient@example.com\r\n"
                    "Subject: Undelivered Mail Returned to Sender\r\n"
                    'Content-Type: multipart/mixed; boundary="boundary-123"\r\n'
                    "\r\n"
                    "--boundary-123\r\n"
                    'Content-Type: multipart/alternative; boundary="boundary-456"\r\n'
                    "\r\n"
                    "--boundary-456\r\n"
                    "Content-Type: text/plain; charset=chinesebig5\r\n"
                    "Content-Transfer-Encoding: quoted-printable\r\n"
                    "\r\n"
                    f"{quopri.encodestring(expected_payload_plain.encode('big5')).decode('big5')}"
                    "\r\n"
                    "--boundary-456\r\n"
                    "Content-Type: text/html; charset=utf-8\r\n"
                    "Content-Transfer-Encoding: base64\r\n"
                    "\r\n"
                    f"{b64encode(expected_payload_html.encode('utf-8')).decode('utf-8')}"
                    "\r\n"
                    "--boundary-456--\r\n"
                    "\r\n"
                    "--boundary-123\r\n"
                    'Content-Type: application/octet-stream; name="attachment.txt"\r\n'
                    "Content-Transfer-Encoding: base64\r\n"
                    'Content-Disposition: attachment; filename="attachment.txt"\r\n'
                    "\r\n"
                    f"{b64encode(b'This is an attachment.').decode('utf-8')}"
                    "\r\n"
                    "--boundary-123--\r\n"
                )
            email = email_from_file(filepath)
        self.assertEqual(
            raw_payload(email), f"{expected_payload_plain}\n{expected_payload_html}"
        )

    def test_payload_dom(self):
        with TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_mail.txt")
            with open(filepath, "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r\n"
                    "To: recipient@example.com\r\n"
                    "Subject: Undelivered Mail Returned to Sender\r\n"
                    "content-type: text/html; charset=utf-8\r\n"
                    "\r\n"
                    "<html>\r\n"
                    "    <body>\r\n"
                    "        <h1>Hello World!</h1>\r\n"
                    "        <p>This is a test.</p>\r\n"
                    "        <a href='http://example.com'>Example</a>\r\n"
                    "    </body>\r\n"
                    "</html>\r\n"
                )
            email = email_from_file(filepath)
        expected = (
            "<html>\n"
            " <body>\n"
            "  <h1>\n"
            "   Hello World!\n"
            "  </h1>\n"
            "  <p>\n"
            "   This is a test.\n"
            "  </p>\n"
            '  <a href="http://example.com">\n'
            "   Example\n"
            "  </a>\n"
            " </body>\n"
            "</html>\n"
        )
        actual = payload_dom(email).prettify()
        self.assertEqual(actual, expected)

    def test_words_from_tokens(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_mail.txt"), "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r\n"
                    "To: recipient@example.com\r\n"
                    "Subject: Undelivered Mail Returned to Sender\r\n"
                    "content-type: text/html; charset=utf-8\r\n"
                    "\r\n"
                    "Hello,   I am under the-water\r\n"
                    "http://under.the-water.com\r\n"
                )
            email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))

        urls, tokens = tokenize_payload(email)
        words = words_from_tokens(tokens)
        expected_urls = {urlparse("http://under.the-water.com")}
        expected_tokens = [
            "Hello,",
            "I",
            "am",
            "under",
            "the-water",
        ]
        expected_words = [
            "Hello",
            "I",
            "am",
            "under",
            "the",
            "water",
        ]
        self.assertEqual(urls, expected_urls)
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(words, expected_words)
