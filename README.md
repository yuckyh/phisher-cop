# phisher-cop

A simple email phishing detector.

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

## Running the project without uv (for the professor's sake)

> Recommended python version: 3.12.11

1.  There is a GitHub action that does this step, you shouldn't have to do it manually.

Create the `requirements.txt` file. This will give pip the packages and versions to install.

```bash
uv export --format requirements.txt --locked --no-dev --no-emit-workspace -o requirements.txt  # We run this command and send requirements.txt to the professor
```

2. Create a virtual environment

```bash
python3 -m venv .venv
```

3. Activate the virtual environment

- On Windows

```bash
.venv\Scripts\activate
```

- On Linux or MacOS
```bash
source .venv/bin/activate
```

4. Install the dependencies

```bash
pip install -r requirements.txt
```

5. Run the scripts

```bash
python3 src/cli.py                 # Run the CLI
python3 src/web.py                 # Run the web server
python3 src/train.py               # Train the ML model
python3 -m unittest tests/main.py  # Run the tests
```
