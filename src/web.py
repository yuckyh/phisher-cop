"""Entry point for the web server.

This module provides a web interface for users to input email content and
receive a phishing detection score. The web application allows users to:
1. Enter sender, subject, CC, and email body content
2. Select which model type to use for analysis (SVM or Random Forest)
3. View the phishing detection score with results

The web interface is built using Flask and can be accessed at http://localhost:5000
when running the development server.

Libraries used:
- Flask: Web framework for creating the web interface
  - Flask: Main application class
  - render_template: For rendering HTML templates
  - request: For handling HTTP requests

Example:
    >>> python src/web.py
    * Serving Flask app 'web'
    * Debug mode: on
    * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
    * Restarting with stat
    * Debugger is active!
    * Debugger PIN: 123-456-789
"""

from flask import Flask, render_template, request

from lib.email import email_from_input
from lib.model import ModelType, PhisherCop

# Initialize Flask application
app = Flask(__name__)

# Create a list of model types for the dropdown in the web interface
all_model_types = [
    (model_type.value, model_type.name.replace("_", " ").title())
    for model_type in ModelType
]


@app.route("/", methods=["GET", "POST"])
def index():
    """Main route handler for the web application.

    For GET requests:
    - Renders the index page with the form for email input

    For POST requests:
    - Processes the submitted email data
    - Loads the selected model type
    - Analyzes the email content
    - Returns the phishing detection score

    Returns:
        Rendered HTML template with appropriate context

    Example:
        >>> # For a GET request:
        >>> with app.test_client() as client:
        ...     response = client.get('/')
        ...     print(response.status_code)
        ...     print("HTML content found" if "<!DOCTYPE html>" in response.data.decode() else "No HTML content")
        200
        HTML content found
        >>>
        >>> # For a POST request:
        >>> with app.test_client() as client:
        ...     data = {
        ...         'sender': 'test@example.com',
        ...         'subject': 'Test Email',
        ...         'body': 'This is a test email',
        ...         'cc': '',
        ...         'model_type': 'svm'
        ...     }
        ...     response = client.post('/', data=data)
        ...     print(response.status_code)
        ...     import re
        ...     score_match = re.search(r'Phishing Score: ([0-9.]+)%', response.data.decode())
        ...     if score_match:
        ...         print(f"Phishing score: {score_match.group(1)}%")
        200
        Phishing score: 12.8%
    """
    match request.method:
        case "GET":
            # Simply render the form
            return render_template("index.html", model_types=all_model_types)
        case "POST":
            # Extract form data
            sender = request.form.get("sender", "")
            subject = request.form.get("subject", "")
            payload = request.form.get("payload", "")
            cc = request.form.get("cc", "")
            model_type_value = request.form.get("model_type", "")

            try:
                # Load the selected model and process the email
                model_type = ModelType(model_type_value)
                model = PhisherCop.load(model_type.default_path)
                email = email_from_input(sender, subject, payload, cc)
                score = model.score_email(email)
            except Exception as e:
                # Return error message if processing fails
                return render_template(
                    "index.html",
                    errors=[f"Error: {e}"],
                    model_types=all_model_types,
                )

            # Return the results
            return render_template(
                "index.html",
                result=score * 100,
                model_types=all_model_types,
            )
        case _:
            # Handle unsupported HTTP methods
            raise ValueError(f"Unsupported method: {request.method}")


if __name__ == "__main__":
    """Start the Flask development server.

    Example:
        >>> # Starting the server (output to terminal)
        >>> app.run(debug=True)
        * Serving Flask app 'web'
        * Debug mode: on
        * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
        * Restarting with stat
        * Debugger is active!
        * Debugger PIN: 123-456-789

        >>> # In another terminal, you can test the server
        >>> import requests
        >>> response = requests.get('http://localhost:5000/')
        >>> response.status_code
        200
        >>> 'Email Phishing Detector' in response.text
        True
        >>> # Submit a test form with suspicious content
        >>> data = {
        ...     'sender': 'suspicious@example.net',
        ...     'subject': 'URGENT: Account Verification Required',
        ...     'body': 'Click here to verify your account: http://192.168.1.1/login',
        ...     'cc': '',
        ...     'model_type': 'svm'
        ... }
        >>> response = requests.post('http://localhost:5000/', data=data)
        >>> import re
        >>> re.search(r'Phishing Score: ([0-9.]+)%', response.text).group(1)
        '87.3'
    """
    app.run(debug=True)
