"""Entry point for the command line interface."""

import click

from lib import MODEL_PATH
from lib.document import (
    email_addresses,
    email_from_file,
    payload_dom,
    tokenize_dom,
    words_from_tokens,
)


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "model_path",
    "-m",
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=MODEL_PATH,
    help="Path to the trained model file.",
)
def main(filepath: str, model_path: str):
    """Reads an email from the file at FILEPATH and prints a confidence score of
    how likely it is to be a phishing email to stdout, along with relevant stats"""
    email = email_from_file(filepath)
    addresses = email_addresses(email)
    dom = payload_dom(email)
    urls, tokens = tokenize_dom(dom)
    words = words_from_tokens(tokens)
    print(f"{model_path=}")
    print(f"{addresses=}")
    print(f"{urls=}")
    print(f"{words=}")
    # TODO: implement the rest


if __name__ == "__main__":
    main()
