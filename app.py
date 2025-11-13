from flask import Flask, render_template, request
import pickle

# Load your model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    transformed_review = vectorizer.transform([review])
    prediction = model.predict(transformed_review)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
