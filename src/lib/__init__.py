"""End-to-end inference for e-mail phishing detection."""

import os
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, cast

from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lib.document import Email, PreprocessedEmail
from lib.domain import Domain, Url
from lib.email_address import EmailAddress
from lib.model import load_model

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[2]
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")

T = TypeVar("T")
R = TypeVar("R")


# TODO: If there is a way to get kwargs from Parallel's kwargs, please change Any
def parallelize(func: Callable[..., R], X: Iterable[object], **kwargs: Any) -> list[R]:
    """Parallelize calls to func over X.

    - If an element of X is a dict, it's passed as kwargs.
    - If it's a tuple/list, it's passed as positional args.
    - Otherwise it's passed as a single positional arg.
    """
    tasks = [
        delayed(func)(**x)
        if isinstance(x, dict)
        else delayed(func)(*x)
        if isinstance(x, (tuple, list))
        else delayed(func)(x)
        # delayed(func)(x)
        for x in X
    ]
    return cast(list[R], Parallel(**kwargs)(tasks))


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
            # "email": email,
        }

    def extract_features(
        self,
        urls: set[Url],
        tokens: list[str],
        words: list[str],
        sender: EmailAddress,
        addresses: list[EmailAddress],
        domains: list[Domain],
    ) -> list[float | str]:
        from .feature_extract import (
            capital_words_ratio,
            count_ip_addresses,
            count_typosquatted_domains,
            count_whitelisted_addresses,
            email_domain_matches_url,
            money_tokens_ratio,
            score_suspicious_words,
        )

        return [
            " ".join(words),
            count_whitelisted_addresses(addresses),
            score_suspicious_words(words),
            count_typosquatted_domains(domains, edit_threshold=1),
            count_ip_addresses(urls),
            email_domain_matches_url(sender, domains),
            capital_words_ratio(words),
            money_tokens_ratio(tokens),
        ]

    def get_pipeline(self) -> Pipeline:
        text_transformer = ColumnTransformer(
            [
                ("tfidf", TfidfVectorizer(), 0),
            ],
            remainder="passthrough",
        )

        return Pipeline(
            [
                ("text", text_transformer),
                ("scaler", StandardScaler(with_mean=False)),  # Standardize features
            ]
        )

    def score_email(self, email: Email) -> float:
        preprocessed_email = self.preprocess_email(email)
        features = self.extract_features(**preprocessed_email)
        ml = load_model(MODEL_PATH)
        return ml.predict([features])[0]
