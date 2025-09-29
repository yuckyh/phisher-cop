"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path
from typing import Callable, Iterable, TypeVar, cast

from joblib import Parallel, delayed

from lib.document import Email, PreprocessedEmail
from lib.domain import Domain

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")

T = TypeVar("T")
R = TypeVar("R")


def parallelize(func: Callable[[T], R], X: Iterable[T]) -> list[R]:
    # Use a list comprehension to avoid generator-related Unknown types and cast the result
    return cast(list[R], Parallel(n_jobs=-1)([delayed(func)(x) for x in X]))


class PhisherCop:
    """High-level inference helper.

    Submodule imports are done inside methods to avoid importing the whole
    package at module import time (which can cause circular imports).
    """

    def preprocess_email(self, email: Email) -> PreprocessedEmail:
        # local imports to avoid circular import during package initialization
        from .document import (
            domains_from_urls,
            get_email_addresses,
            tokenize_payload,
            words_from_tokens,
        )
        from .email_address import parse_email_address

        urls, tokens = tokenize_payload(email)
        words = words_from_tokens(tokens)
        return {
            "urls": urls,
            "tokens": tokens,
            "words": words,
            "sender": parse_email_address(str(email["From"])),
            "addresses": get_email_addresses(email),
            "domains": domains_from_urls(urls),
            "email": email,
        }

    def extract_features(self, email: PreprocessedEmail) -> list[float | str | list[Domain]]:  # noqa: F821
        from .feature_extract import (
            capital_words_ratio,
            count_ip_addresses,
            count_whitelisted_addresses,
            email_domain_matches_url,
            money_tokens_ratio,
            score_suspicious_words,
        )

        urls = email["urls"]
        tokens = email["tokens"]
        words = email["words"]
        sender = email["sender"]
        addresses = email["addresses"]
        domains = email["domains"]

        return [
            " ".join(words),
            count_whitelisted_addresses(addresses),
            score_suspicious_words(words),
            domains,
            count_ip_addresses(urls),
            email_domain_matches_url(sender, domains),
            capital_words_ratio(words),
            money_tokens_ratio(tokens),
        ]

    def score_email(self, email: str) -> float:
        raise Exception("TODO: Not implemented")
