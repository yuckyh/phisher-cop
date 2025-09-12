from domain import Domain


def get_words(document: str) -> list[str]:
    # 1. sanitize
    # 2. split by spaces
    # 3. ???
    # 4. profit
    raise NotImplementedError


def caps_count(words: list[str]) -> int:
    return sum(1 for word in words for char in word if char.isupper())


def get_urls(words: list[str]) -> list[Domain]:
    raise NotImplementedError
