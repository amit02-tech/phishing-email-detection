from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    email_text = ""

    if request.method == 'POST':
        email_text = request.form.get('email_content')
        vectorized_text = vectorizer.transform([email_text])
        result = model.predict(vectorized_text)
        prediction = "PHISHING" if result[0] == 1 else "SAFE"

    return render_template('index.html', prediction=prediction, email_text=email_text)


if __name__ == '__main__':
    app.run(debug=True)