#!/usr/bin/env bash
set -e                                                    # Exit on error
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null  # cd to directory of this script

uv sync                                                # Ensure dependencies are up-to-date
uv run coverage run -m unittest tests/main.py || true  # Run tests, ignoring errors due to test failures
uv run coverage report --format=markdown -m            # Display coverage report
