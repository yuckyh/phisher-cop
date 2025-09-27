import os
import unittest
from tempfile import TemporaryDirectory

from src.lib.document import email_from_file, payload_dom, raw_payload, tokenize_payload


class TestDocument(unittest.TestCase):
    def test_email_from_file(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_mail.txt"), "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r"
                    "To: recipient@example.com\r"
                    "Content-Type: text/plain; charset=latin-1\r"
                    "Subject: Undelivered Mail Returned to Sender\r"
                    "\r"
                    "This is the mail system at host example.com. I’m sorry to have to inform you that your message could not be delivered to one or more recipients. It’s attached below.\r"
                    "\r"
                    "--- Original message ---\r"
                )
            email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        self.assertEqual(email.get_content_type(), "text/plain")
        self.assertEqual(email.get_content_charset(), "latin-1")
        self.assertEqual(
            email["From"], "Mail Delivery Subsystem <postmaster@example.com>"
        )
        self.assertEqual(email["To"], "recipient@example.com")
        self.assertEqual(email["Subject"], "Undelivered Mail Returned to Sender")
        self.assertIn("This is the mail system at host", email.get_payload())

    def test_raw_payload(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_mail.txt"), "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r"
                    "To: recipient@example.com\r"
                    "Subject: Undelivered Mail Returned to Sender\r"
                    "This is the mail system at host example.com. Im sorry to have to inform you that your message could not be delivered to one or more recipients. Its attached below.\r"
                    "--- Original message ---\r"
                )
            email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        payload = raw_payload(email)
        self.assertIn("This is the mail system at host", payload)
        self.assertIn("Its attached below.", payload)
        self.assertIn("--- Original message ---", payload)

    def test_payload_dom(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_mail.txt"), "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r"
                    "To: recipient@example.com\r"
                    "Subject: Undelivered Mail Returned to Sender\r"
                    "<html>"
                    "    <body>"
                    "        <h1>Hello World!</h1>"
                    "        <p>This is a test.</p>"
                    "        <a href='http://example.com'>Example</a>"
                    "    </body>"
                    "</html>"
                )
            email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        payload = payload_dom(email)
        expected_payload = '<html> <body> <h1>Hello World!</h1> <p>This is a test.</p> <a href="http://example.com">Example</a> </body></html>'
        self.assertEqual(str(payload), expected_payload)

    def test_tokenize_payload(self):
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_mail.txt"), "w") as f:
                f.write(
                    "From: Mail Delivery Subsystem <postmaster@example.com>\r"
                    "To: recipient@example.com\r"
                    "Subject: Undelivered Mail Returned to Sender\r"
                    "<html>"
                    "    <body>"
                    "        <h1>Hello World!</h1>"
                    "        <p>This is a test.</p>"
                    "        <a href='http://example.com'>Example</a>"
                    "    </body>"
                    "</html>"
                )
            email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        _, tokens = tokenize_payload(email)
        expected_tokens = [
            "Hello",
            "World!",
            "This",
            "is",
            "a",
            "test.",
            "Example",
        ]
        self.assertEqual(tokens, expected_tokens)
