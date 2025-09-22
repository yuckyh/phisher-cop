"""Entry point for the command line interface."""

import os

import click


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, readable=True))
def main(filepath: str):
    """Reads an email from the file at FILEPATH and prints a confidence score of
    how likely it is to be a phishing email to stdout, along with relevant stats"""
    filepath = os.path.realpath(filepath)
    print(f"{filepath=}")
    # TODO: implement the rest


if __name__ == "__main__":
    main()
