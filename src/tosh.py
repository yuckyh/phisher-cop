# Just starting out on a coding project for the first time, don't sue me guys ;-;

r'''
Main objective here:
Keyword Detection and Scoring
1. Scan email subject and body for suspicious keywords (e.g., 'urgent', 'verify', 'account')
2. Assign higher risk scores for suspicious keywords appearing in subject lines or early in the message
3. Recommended to get the list of suspicious words from a public source online (done)
4. Should return Dict[str, int], where each key is the suspicious word, and each value is the minimum index of the sus word in the email.
'''

# The journey of 10,000 lines of code begins with a few libraries... 
# import flask
# from flask import Flask, render_template, request
from bs4 import BeautifulSoup
# import numpy as np
import re
# from sklearn.feature.extraction.text import TfidVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# import tldextract

SUSPICIOUS_KEYWORDS = [
    'urgent',
    'hurry',
    'verify',
    'account',
    'action required',
    'password',
    'login',
    'security alert',
    'immediate',
    'invoice',
    'payment',
    'click here',
    'link',
    'update',
    'confirm',
    'free'
]

def score_email_keywords(subject: str, body: str) -> dict[str, int]:
    """
    Scans the email subject and body for suspicious keywords and assigns a risk score based on their location.

    Parameters:
        subject (str): The subject line of the email.
        body (str): The HTML or plain text body of the email.

    Returns:
        dict[str, int]: A dictionary where each key is a suspicious keyword found, and each value is the assigned risk score.
    """
    # 1. Extract plain text from the body using BeautifulSoup
    soup = BeautifulSoup(body, 'html.parser')
    plain_body = soup.get_text()

    # 2. Combine and normalize the text for searching
    # We use a unique separator to differentiate subject and body indices
    full_text = (subject.lower() + " [SEP] " + plain_body.lower())
    indices = {}

    # 3. Iterate through keywords and find their occurrences
    for keyword in SUSPICIOUS_KEYWORDS:
        # Use a regex pattern for accurate, whole-word, case-insensitive matching
        # \b ensures we match word boundaries (e.g., 'free' but not 'freeway')
        pattern = r'\b' + re.escape(keyword) + r'\b'
        match = re.search(pattern, full_text)

        if match:
            # Get the starting index of the first match
            min_index = match.start()
            # Store the minimum index for this keyword
            indices[keyword] = min_index

    return indices

def extract_suspicious_words():
    pass