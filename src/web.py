"""Entry point for the web server."""

from flask import Flask, render_template_string, request

from lib.document import email_from_input, payload_dom, tokenize_dom, words_from_tokens
from lib.model import load_model

# creating Flask app instance
app = Flask(__name__)


# route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get data from the form
        sender = request.form.get("sender", "")
        subject = request.form.get("subject", "")
        payload = request.form.get("payload", "")

        # Use the actual document.py function to create the email object
        email = email_from_input(
            sender=sender, recipient="", cc="", subject=subject, payload=payload
        )

        # Process the email body to get features for the model
        dom = payload_dom(email)
        tokens = tokenize_dom(dom)
        words = words_from_tokens(tokens)

        # Load the trained model and make a prediction
        model = load_model()
        prediction = model.predict([words])[0]

        # Map the numeric prediction to a human-readable string
        result = "Phishing" if prediction == 1 else "Not Phishing"

        # Render the template with the result
        return render_template_string(HTML_TEMPLATE, result=result)

    # For GET requests, show the empty form
    return render_template_string(HTML_TEMPLATE, result=None)


if __name__ == "__main__":
    app.run(debug=True)
