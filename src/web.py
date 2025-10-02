"""Entry point for the web server."""

from flask import Flask, render_template, request

from lib import MODEL_PATH
from lib.email import email_from_input
from lib.model import PhisherCop

app = Flask(__name__)


model = PhisherCop.load(MODEL_PATH)


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
                email = email_from_input(sender, subject, payload, cc)
                score = model.score_email(email)
            except Exception as e:
                return render_template("index.html", errors=[f"Error: {e}"])
            return render_template("index.html", result=score)
        case _:
            raise ValueError(f"Unsupported method: {request.method}")


if __name__ == "__main__":
    app.run(debug=True)
