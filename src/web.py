"""Entry point for the web server."""

from hashlib import sha256

from flask import Flask, render_template, request

from lib.email import Email, PreprocessedEmail, email_from_input
from lib.feature_extract import extract_features

app = Flask(__name__)


# TODO: replace this with the actual model
class DummyModel:
    def predict(self, email: Email) -> float:
        preprocessed = PreprocessedEmail(email, ignore_errors=False)
        features = extract_features(preprocessed)
        # Get a random float in [0, 1]
        float_bytes = sha256(str(features).encode("utf-8")).digest()[0:4]
        return int.from_bytes(float_bytes, "big") / (2**32 - 1)


model = DummyModel()


@app.route("/", methods=["GET", "POST"])
def index():
    match request.method:
        case "GET":
            return render_template("index.html")
        case "POST":
            sender = request.form.get("sender", "")
            recipient = request.form.get("recipient", "")
            subject = request.form.get("subject", "")
            payload = request.form.get("payload", "")
            cc = request.form.get("cc", "")

            try:
                email = email_from_input(sender, recipient, subject, payload, cc)
                score = model.predict(email)
            except Exception as e:
                return render_template("index.html", errors=[f"Error: {e}"])

            return render_template("index.html", result=score)
        case _:
            raise ValueError(f"Unsupported method: {request.method}")


if __name__ == "__main__":
    app.run(debug=True)
