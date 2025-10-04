# phisher-cop

A simple email phishing detector.

## Message to Professor

Hi Professor! If you're reading this, please skip ahead to the "Running the project without uv" section.
All the necessary files that aren't tracked by git are included in this zip file we submitted, everything should just work out-of-the-box.

The `.git/` folder is also included as a proof of work.
You can see the full commit history by running `git log -p`.

## Table Of Contents

- [Development](#development)
  - [Linting and formatting](#linting-and-formatting)
  - [GitHub](#github)
- [Training the model](#training-the-model)
- [Running the project without uv](#running-the-project-without-uv)

## Development

This project uses uv to manage the Python environment.

You can install uv [here](https://docs.astral.sh/uv/getting-started/installation/).

To run a script within the uv environment, use:

```bash
uv run -s <script-name>.py  # 'src/cli.py', 'src/web.py' or 'src/train.py'
```

To add a project dependency, run:

```bash
uv add <package-name>
```

To remove a project dependency, run:

```bash
uv remove <package-name>
```

To run a single test file, run:

```bash
uv run python -m unittest tests/<file-name>.py
```

To run all tests and generate a coverage report, run:

- On Linux or MacOS
```bash
./test.sh
```

- On Windows
```powershell
sh .\test.sh  # If you already have git installed, this will execute the script with Git Bash
```

### Linting and formatting

If you're on VS Code, install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

Ruff has already been configured in the `pyproject.toml`, so you do not need to configure anything on your side.

However, I highly recommend that you turn on "format on save" in your VS Code settings to make full use of Ruff.

[Error Lens](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens) is another great extension to have.

### GitHub

Make a separate branch for each feature or bugfix you work on. When you are done working on your branch, don't forget to make a pull request and ping someone for review.

## Training the model

Download [the dataset](https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus) as a zip and place it in this project's root directory.
The file should be named `archive.zip`.

Then, just run the training script:

```bash
uv run -s src/train.py
```

Note that this only trains a single model. To change the type of model trained, modify the `MODEL_TYPE` variable in `src/train.py`.

## Running the project without uv

> Recommended python version: 3.12.11

1. Create a virtual environment

```bash
python3 -m venv .venv
```

2. Activate the virtual environment

- On Windows

```bash
.venv\Scripts\activate
```

- On Linux or MacOS
```bash
source .venv/bin/activate
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Run the scripts

```bash
python3 src/cli.py         # Run the CLI
python3 src/web.py         # Run the web server
python3 src/train.py       # Train the ML model
python3 -m unittest tests  # Run the tests
```

Running `src/train.py` will also unpack the dataset to `data/`. Each file in `data/` is an email that can be used as input to the CLI.
