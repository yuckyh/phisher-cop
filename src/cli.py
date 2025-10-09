"""
Entry point for the command line interface.

This module provides a command-line interface for analyzing email files
to determine if they are phishing attempts. It uses the trained PhisherCop
model to score emails and provides feedback on whether they are likely
legitimate or phishing emails.

Usage:
    python cli.py [OPTIONS] FILEPATH

Examples:
    >>> python cli.py data/test/spam/0001.txt
    Phishing score: 85.23%
    This email is likely a phishing email.

    >>> python cli.py --model-path models/random_forest.joblib data/test/ham/0001.txt
    Phishing score: 12.45%
    This email is likely a legitimate email.
"""

import click

from lib.email import email_from_file
from lib.model import ModelType, PhisherCop


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "model_path",
    "-m",
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=ModelType.SVM.default_path,
    help="Path to the trained model file.",
)
def main(filepath: str, model_path: str):
    # Load the model from the specified path
    model = PhisherCop.load(model_path)

    # Parse the email file
    email = email_from_file(filepath)

    # Calculate the phishing score
    score = model.score_email(email)

    # Print the results
    print(f"Phishing score: {score * 100:.2f}%")
    if score >= 0.5:
        print("This email is likely a phishing email.")
    else:
        print("This email is likely a legitimate email.")


if __name__ == "__main__":
    main()
