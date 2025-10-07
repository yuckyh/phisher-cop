#!/usr/bin/env bash
set -e                                                    # Exit on error
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null  # cd to directory of this script

uv sync                                        # Ensure dependencies are up-to-date
rm -f ./.coverage                              # Remove old coverage data
uv run coverage run -m unittest tests || true  # Run tests, ignoring errors due to test failures
uv run coverage report --format=markdown -m    # Display coverage report
