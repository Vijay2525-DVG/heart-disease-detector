from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print(f"Model expects features: {model.feature_names_in_}")

@app.route("/")
def home():
    return "Heart Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Create DataFrame with correct column names (matching training)
        features_dict = {
            'age': data["age"],
            'sex': data["sex"],
            'cp': data["cp"],
            'trtbps': data["trestbps"],
            'chol': data["chol"],
            'fbs': data["fbs"],
            'restecg': data["restecg"],
            'thalachh': data["thalach"],
            'exng': data["exang"],
            'oldpeak': data["oldpeak"],
            'slp': data["slope"],
            'caa': data["ca"],
            'thall': data["thal"]
        }
        
        features = pd.DataFrame([features_dict])
        
        # Get prediction and probability
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        disease_probability = probability[1]
        
        print(f"\nInput: Age={data['age']}, Sex={data['sex']}, CP={data['cp']}")
        print(f"Prediction: {prediction} (0=No Disease, 1=Disease)")
        print(f"Probabilities: No Disease={probability[0]:.3f}, Disease={probability[1]:.3f}")
        
        # Determine risk level based on probability with calibrated thresholds
        # Based on your model's actual behavior
        if disease_probability >= 0.70:
            result = "High Risk - Heart Disease Detected"
            risk_level = "high"
        elif disease_probability >= 0.40:
            result = "Moderate Risk - Heart Disease Possible"
            risk_level = "moderate"
        else:
            result = "Low Risk - No Heart Disease Detected"
            risk_level = "low"
        
        print(f"Result: {result}")
        print(f"Risk Level: {risk_level}")
        
        return jsonify({
            "prediction": int(prediction),
            "result": result,
            "risk_level": risk_level,
            "probability": float(disease_probability),
            "probabilities": {
                "no_disease": float(probability[0]),
                "disease": float(probability[1])
            }
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/ui")
def ui():
    return app.send_static_file("index.html")