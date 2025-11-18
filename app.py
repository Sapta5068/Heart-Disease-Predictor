from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the heart disease model and scaler
try:
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model files not found!")
    print("Please run 'python train_heart_model.py' first to create the model files.")
    model = None
    scaler = None

# ============== ROUTES ==============

@app.route('/')
def home():
    """Homepage - Landing page with information"""
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    """Prediction form page"""
    if model is None:
        return "Model not loaded. Please train the model first by running 'python train_heart_model.py'", 500
    return render_template('predict.html')

# ============== API ENDPOINT ==============

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for heart disease prediction"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
            
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['restecg']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['slope']),
            float(data['ca']),
            float(data['thal'])
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_disease': float(probability[0] * 100),
                'disease': float(probability[1] * 100)
            },
            'message': 'High risk of heart disease detected' if prediction == 1 else 'Low risk of heart disease',
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
        }
        
        return jsonify(result)
    
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value provided: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# ============== ERROR HANDLERS ==============

@app.errorhandler(404)
def not_found(e):
    return "<h1>404 - Page Not Found</h1><p>The page you're looking for doesn't exist.</p>", 404

@app.errorhandler(500)
def server_error(e):
    return "<h1>500 - Server Error</h1><p>Something went wrong on the server.</p>", 500

# ============== MAIN ==============

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏥 HEART DISEASE PREDICTION SYSTEM")
    print("="*50)
    
    if model is not None:
        print("✓ Flask server starting...")
        print("✓ Open your browser and go to: http://127.0.0.1:5000/")
        print("="*50 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start server - model files missing!")
        print("\nPlease run this command first:")
        print("   python train_heart_model.py")
        print("="*50 + "\n")