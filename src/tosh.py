# The journey of 10,000 lines of code begins with a few libraries...

import re

from bs4 import BeautifulSoup

from lib.dataset import load_data
from lib.document import (
    email_addresses,
    email_from_file,
    payload_dom,
    tokenize_dom,
    words_from_tokens,
)

if __name__ == "__main__":
    load_data()

    email = email_from_file("data/train/spam/0000.txt")
    addresses = email_addresses(email)
    dom = payload_dom(email)
    urls, tokens = tokenize_dom(dom)
    words = words_from_tokens(tokens)
    print(addresses, urls, words)

r"""
Main objective here:
Keyword Detection and Scoring
1. Scan email subject and body for suspicious keywords (e.g., 'urgent', 'verify', 'account')
2. Assign higher risk scores for suspicious keywords appearing in subject lines or early in the message
3. Should return Dict[str, int], where each key is the suspicious word, and each value is the minimum index of the sus word in the email.
"""

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "hurry",
    "verify",
    "account",
    "action required",
    "password",
    "login",
    "security alert",
    "suspicious activity",
    "limited time",
    "low price",
    "immediate",
    "invoice",
    "payment",
    "click here",
    "link",
    "update",
    "confirm",
    "free",
]


# Time to make the actual function
def score_email_keywords(subject: str, body: str) -> dict[str, tuple[str, int]]:
    """
    Scans the email subject and body for suspicious keywords and returns the minimum index and location of each found keyword.

    Parameters:
        subject (str): The subject line of the email.
        body (str): The HTML or plain text body of the email.

     Returns:
        dict[str, tuple[str, int]]: A dictionary where each key is a suspicious keyword found, and each value is a tuple (location, index), where location is 'subject' or 'body'.
    """
    soup = BeautifulSoup(body, "html.parser")
    plain_body = soup.get_text()

    subject_lower = subject.lower()
    body_lower = plain_body.lower()
    sep = " [SEP] "
    sep_index = len(subject)
    full_text = subject_lower + sep + body_lower
    indices = {}

    for keyword in SUSPICIOUS_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        match = re.search(pattern, full_text)
        if match:
            min_index = match.start()
            sep_index = len(subject_lower)
            if min_index < sep_index:
                location = "subject"
                rel_index = min_index
            elif min_index > sep_index + len(sep) - 1:
                location = "body"
                rel_index = min_index - (sep_index + len(sep))
            else:
                location = "separator"
                rel_index = 0
            indices[keyword] = (location, rel_index)

    return indices


score_email_keywords(email)
