from flask import Flask, render_template, request
import pickle, os

# Get absolute paths (important for Render)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model.pkl')
vectorizer_path = os.path.join(base_dir, 'tfidf.pkl')  # or 'vectorizer.pkl'

# Load model and vectorizer in binary mode
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

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
