from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    print("✅ Models loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Features: {feature_names}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please run training first.'}), 500
    
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
            print("Received JSON:", data)
        else:
            # Try to get from form data
            data = {
                'recency': request.form.get('recency'),
                'frequency': request.form.get('frequency'),
                'monetary': request.form.get('monetary'),
                'time': request.form.get('time')
            }
            print("Received form data:", data)
        
        # Extract values
        recency = float(data.get('recency', 0))
        frequency = float(data.get('frequency', 0))
        monetary = float(data.get('monetary', 0))
        time = float(data.get('time', 0))
        
        print(f"Parsed values - Recency: {recency}, Frequency: {frequency}, Monetary: {monetary}, Time: {time}")
        
        # Validate input
        if recency < 0 or frequency < 0 or monetary < 0 or time < 0:
            return jsonify({'error': 'All values must be non-negative'}), 400
        
        # Feature engineering
        freq_monetary = frequency * monetary
        recency_freq = recency * frequency
        donation_rate = frequency / (time + 1) if time >= 0 else 0
        monetary_rate = monetary / (time + 1) if time >= 0 else 0
        
        # Create feature array
        features = np.array([[recency, frequency, monetary, time, 
                             freq_monetary, recency_freq, donation_rate, monetary_rate]])
        
        print(f"Features: {features}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        print(f"Scaled features: {features_scaled}")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        print(f"Prediction: {prediction}")
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features_scaled)[0][1]
        else:
            probability = 0.5
        
        result = {
            'will_donate': bool(int(prediction)),
            'probability': float(probability),
            'message': '✅ This person is likely to donate blood' if prediction == 1 else '❌ This person is unlikely to donate blood',
            'confidence': f"{probability*100:.1f}%"
        }
        
        print(f"Result: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please run the training script first.'}), 500
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        print(f"Batch data shape: {df.shape}")
        print(f"Batch columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_cols = ['Recency', 'Frequency', 'Monetary', 'Time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400
        
        # Feature engineering
        df['Freq_Monetary'] = df['Frequency'] * df['Monetary']
        df['Recency_Freq'] = df['Recency'] * df['Frequency']
        df['Donation_Rate'] = df['Frequency'] / (df['Time'] + 1)
        df['Monetary_Rate'] = df['Monetary'] / (df['Time'] + 1)
        
        # Select features
        X = df[feature_names]
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = np.full(len(predictions), 0.5)
        
        # Add predictions to dataframe
        df['Prediction'] = ['Will Donate' if p == 1 else "Won't Donate" for p in predictions]
        df['Probability'] = probabilities
        
        # Summary
        summary = {
            'total': len(df),
            'will_donate': int(sum(predictions)),
            'wont_donate': int(len(df) - sum(predictions)),
            'donation_rate': f"{sum(predictions)/len(df)*100:.1f}%"
        }
        
        # Convert to JSON
        results = df[required_cols + ['Prediction', 'Probability']].to_dict('records')
        
        return jsonify({
            'summary': summary,
            'results': results
        })
    
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'n_features': len(feature_names)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)