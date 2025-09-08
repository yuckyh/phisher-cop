from enum import Enum

from src.email import Email


class Feature(Enum):
    EMAIL_DOMAIN = 1
    KEYWORDS = 2
    EARLY_KEYWORDS = 3


# Not sure about return type, feel free to change it
def extract_features(email: Email, features: list[Feature]) -> dict[Feature, float]:
    """Extract the list of features in `features` from `email`."""
    raise Exception("TODO: Not implemented")


# Alternatively, to make full use of vectorisation:
# def extract_features(email: list[Email], features: list[Feature]) -> dict[Feature, float]:
