import os
import unittest
from tempfile import TemporaryDirectory

from src.lib.document import email_from_file, payload_dom, raw_payload, tokenize_dom


class TestDocument(unittest.TestCase):
    def test_email_from_file(self):
        with TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "test_mail.txt"), "w", encoding="latin-1"
            ) as f:
                f.write(
                    """
From: Mail Delivery Subsystem <postmaster@example.com>
To: recipient@example.com
Subject: Undelivered Mail Returned to Sender

This is the mail system at host example.com. I’m sorry to have to inform you that your message could not be delivered to one or more recipients. It’s attached below.

--- Original message ---

"""
                )
        email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        self.assertEqual(
            email["From"], "Mail Delivery Subsystem <postmaster@example.com>"
        )
        self.assertEqual(email["To"], "recipient@example.com")
        self.assertEqual(email["Subject"], "Undelivered Mail Returned to Sender")
        self.assertIn("This is the mail system at host", email.get_payload())

    def test_raw_payload(self):
        with TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "test_mail.txt"), "w", encoding="latin-1"
            ) as f:
                f.write(
                    """
From: Mail Delivery Subsystem <postmaster@example.com>
To: recipient@example.com
Subject: Undelivered Mail Returned to Sender
This is the mail system at host example.com. I’m sorry to have to inform you that your message could not be delivered to one or more recipients. It’s attached below.
--- Original message ---
"""
                )
        email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        payload = raw_payload(email)
        self.assertIn("This is the mail system at host", payload)
        self.assertIn("It’s attached below.", payload)
        self.assertIn("--- Original message ---", payload)

    def test_payload_dom(self):
        with TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "test_mail.txt"), "w", encoding="latin-1"
            ) as f:
                f.write(
                    """
From: Mail Delivery Subsystem <postmaster@example.com>
To: recipient@example.com
Subject: Undelivered Mail Returned to Sender
<html>
    <body>
        <h1>Hello World!</h1>
        <p>This is a test.</p>
        <a href="http://example.com">Example</a>
    </body>
</html>
"""
                )
        email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        payload = payload_dom(email)
        expected_payload = "Hello World! This is a test. Example"
        self.assertEqual(payload, expected_payload)

    def test_tokenize_dom(self):
        with TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "test_mail.txt"), "w", encoding="latin-1"
            ) as f:
                f.write(
                    """
From: Mail Delivery Subsystem <postmaster@example.com>
To: recipient@example.com
Subject: Undelivered Mail Returned to Sender
<html>
    <body>
        <h1>Hello World!</h1>
        <p>This is a test.</p>
        <a href="http://example.com">Example</a>
    </body>
</html>
"""
                )
        email = email_from_file(os.path.join(tmpdir, "test_mail.txt"))
        dom = payload_dom(email)
        _, tokens = tokenize_dom(dom)
        expected_tokens = [
            "html",
            "body",
            "h1",
            "Hello",
            "World",
            "p",
            "This",
            "is",
            "a",
            "test",
            "a",
            "href",
            "http",
            "example",
            "com",
            "Example",
        ]
        self.assertEqual(tokens, expected_tokens)
