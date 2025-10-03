"""Entry point for the web server."""

from flask import Flask, render_template, request

from lib.email import email_from_input
from lib.model import ModelType, PhisherCop

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    match request.method:
        case "GET":
            return render_template("index.html")
        case "POST":
            sender = request.form.get("sender", "")
            subject = request.form.get("subject", "")
            payload = request.form.get("payload", "")
            cc = request.form.get("cc", "")

            try:
                model = PhisherCop.load(ModelType.RANDOM_FOREST.default_path)
                email = email_from_input(sender, subject, payload, cc)
                score = model.score_email(email)
            except Exception as e:
                return render_template("index.html", errors=[f"Error: {e}"])
            return render_template("index.html", result=score * 100)
        case _:
            raise ValueError(f"Unsupported method: {request.method}")


if __name__ == "__main__":
    app.run(debug=True)
