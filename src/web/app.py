"""Entry point for the web server."""

import os
import sys

from flask import Flask, render_template, request

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# importinh the functions to process data
from lib.document import (  # noqa: E402
    email_from_input,
    payload_dom,
    tokenize_dom,
    words_from_tokens,
)
from lib.model import load_model  # noqa: E402

# creating Flask app instance
app = Flask(__name__)

"""TODO #18:
input validation
input sanitisation
provide user feedback (error messages)"""


# route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    """Processes the data after receiving from the Form and
    returns the result of analysis"""
    if request.method == "POST":
        # getting data from the form
        sender = request.form.get("sender", "")
        recipient = request.form.get("recipient", "")
        cc = request.form.get("cc", "")
        subject = request.form.get("subject", "")
        payload = request.form.get("payload", "")

        # creating email object
        email = email_from_input(
            sender=sender, recipient=recipient, cc=cc, subject=subject, payload=payload
        )

        # Process the email body to get features for the model
        dom = payload_dom(email)
        tokens = tokenize_dom(dom)
        words = words_from_tokens(tokens)

        # Load the trained model and make a prediction
        model = load_model()
        prediction = model.predict([words])[0]

        # TEMPORARY
        # Map the numeric prediction to a human-readable string
        result = "Phishing" if prediction == 1 else "Not Phishing"

        # Render the template with the result
        return render_template("index.html", result=result)

    # For GET requests, show the empty form (no result)
    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
