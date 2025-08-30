from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return "Random Forest Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get input data as JSON
        features = np.array(data["features"]).reshape(1, -1)  # Convert to NumPy array
        prediction = model.predict(features)[0]  # Get prediction
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
