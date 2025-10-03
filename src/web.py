"""Entry point for the web server."""

from flask import Flask, render_template, request

from lib.email import email_from_input
from lib.model import ModelType, PhisherCop

app = Flask(__name__)
all_model_types = [
    (model_type.value, model_type.name.replace("_", " ").title())
    for model_type in ModelType
]


@app.route("/", methods=["GET", "POST"])
def index():
    match request.method:
        case "GET":
            return render_template("index.html", model_types=all_model_types)
        case "POST":
            sender = request.form.get("sender", "")
            subject = request.form.get("subject", "")
            payload = request.form.get("payload", "")
            cc = request.form.get("cc", "")
            model_type_value = request.form.get("model_type", "")

            try:
                model_type = ModelType(model_type_value)
                model = PhisherCop.load(model_type.default_path)
                email = email_from_input(sender, subject, payload, cc)
                score = model.score_email(email)
            except Exception as e:
                return render_template(
                    "index.html",
                    errors=[f"Error: {e}"],
                    model_types=all_model_types,
                )
            return render_template(
                "index.html",
                result=score * 100,
                model_types=all_model_types,
            )
        case _:
            raise ValueError(f"Unsupported method: {request.method}")


if __name__ == "__main__":
    app.run(debug=True)
