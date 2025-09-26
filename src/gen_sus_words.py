"""Generate the list of suspicious words from the training split."""

from lib.dataset import load_data
from lib.document import payload_dom, tokenize_dom, words_from_tokens


def top_n(word_counts: dict[str, int], n: int) -> dict[str, int]:
    descending = sorted(
        word_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {k: v for k, v in descending[:n]}


if __name__ == "__main__":
    # Only use the training split to avoid data leakage
    train, _, _ = load_data()
    emails, labels = train

    ham_word_counts = {}
    spam_word_counts = {}
    for email, label in zip(emails, labels):
        dom = payload_dom(email)
        _, tokens = tokenize_dom(dom)
        words = words_from_tokens(tokens)
        word_counts = ham_word_counts if label == 0 else spam_word_counts
        for word in words:
            word = word.lower().strip()
            if not word or not word.isalpha():
                continue
            word_counts[word] = word_counts.get(word, 0) + 1

    # Remove common "ham" words from "spam" words
    ham = top_n(ham_word_counts, 80)
    spam = top_n(spam_word_counts, 80)
    for word in ham.keys():
        if word in spam:
            del spam[word]

    print("ham:", ham)
    print("spam:", spam)
    print(
        "Clean set to copy and paste:",
        {word for word in spam.keys() if len(word) >= 4},
    )
