# Blood Donor Prediction System

A machine learning web application that predicts whether a person will donate blood based on their donation history. Built with Flask and scikit-learn, deployed on Render.

## 🚀 Live Demo
[https://blood-donor-prediction-fvts.onrender.com/](https://blood-donor-prediction-fvts.onrender.com/)

## ✨ Features

- **Single Prediction**: Enter donation history details to get an instant prediction
- **Batch Prediction**: Upload a CSV file to predict for multiple donors at once
- **Probability Visualization**: See the confidence level of each prediction with a visual indicator
- **Multiple ML Models**: Uses 8 different algorithms and automatically selects the best one
- **Interactive Web Interface**: Clean, responsive UI built with Bootstrap

## 🛠️ Tech Stack

- **Backend**: Flask, Python 3.12
- **Machine Learning**: scikit-learn, pandas, numpy, XGBoost
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Deployment**: Render

## 📁 Project Structure

```
blood-donor-prediction/
├── app.py                 # Flask application
├── blood_donor_analysis.py # Model training script
├── requirements.txt       # Python dependencies
├── .python-version        # Python version for Render
├── templates/
│   └── index.html        # Web interface
├── models/               # Saved models (created after training)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
└── README.md
```

## 🚦 How It Works

1. **Data Preprocessing**: The system uses features like recency, frequency, monetary value, and time since first donation
2. **Feature Engineering**: Creates interaction features (`Freq_Monetary`, `Recency_Freq`, `Donation_Rate`, `Monetary_Rate`)
3. **Model Training**: Trains 8 algorithms and selects the best performing one
4. **Prediction**: Makes predictions with confidence scores

## 📊 Input Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| Recency | Months since last donation | 2 |
| Frequency | Total number of donations | 50 |
| Monetary | Total blood donated (c.c.) | 12500 |
| Time | Months since first donation | 98 |

## 🧪 Test Data

You can test with these sample values (they should return "Likely to Donate"):
- Recency: 2, Frequency: 50, Monetary: 12500, Time: 98
- Recency: 0, Frequency: 13, Monetary: 3250, Time: 28
- Recency: 4, Frequency: 4, Monetary: 1000, Time: 4

## 🚀 Local Development

### Prerequisites
- Python 3.12 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/vibamalika25/Blood-Donor-Prediction.git
cd Blood-Donor-Prediction

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models (creates models/ directory)
python blood_donor_analysis.py

# 5. Run the Flask application
python app.py

# 6. Open browser and go to http://localhost:5000
```

## 📤 Deployment to Render

### Option 1: One-Click Deploy (If your repo is public)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Option 2: Manual Deployment
1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: Specify 3.12 in `.python-version`
6. Click "Create Web Service"

### Important Files for Deployment
- **`.python-version`**: Contains `3.12.8`
- **`requirements.txt`**: All Python dependencies
- **`app.py`**: Must use the PORT environment variable:
  ```python
  port = int(os.environ.get('PORT', 5000))
  app.run(host='0.0.0.0', port=port)
  ```

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost | ~79% | ~85% |
| Random Forest | ~78% | ~84% |
| Gradient Boosting | ~78% | ~83% |
| Logistic Regression | ~76% | ~81% |
| SVM | ~75% | ~80% |

*Note: Actual performance may vary slightly based on the test split.*

## 🐛 Troubleshooting

### Build Failures on Render
- **"No such file: requirements.txt"**: Ensure the file exists in your repository root
- **"Cython compilation fails"**: Use Python 3.12 (create `.python-version` file)
- **"Module not found"**: Update your `requirements.txt` with compatible versions

### Common Issues
1. **Models not found**: Run `python blood_donor_analysis.py` first
2. **Port conflicts**: Change the port in `app.py`
3. **File upload errors**: Ensure CSV has columns: `Recency, Frequency, Monetary, Time`

## 📄 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

**Vibamalika** - [GitHub](https://github.com/vibamalika25)

For issues or questions:
- Open an issue on GitHub
- Check Render deployment logs for server errors
