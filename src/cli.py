"""Entry point for the command line interface."""

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
    """Reads an email from the file at FILEPATH and prints a confidence score of
    how likely it is to be a phishing email to stdout, along with relevant stats"""
    model = PhisherCop.load(model_path)
    email = email_from_file(filepath)
    score = model.score_email(email)
    print(f"Phishing score: {score * 100:.2f}%")
    if score >= 0.5:
        print("This email is likely a phishing email.")
    else:
        print("This email is likely a legitimate email.")


if __name__ == "__main__":
    main()
