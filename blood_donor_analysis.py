import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

# Create models directory
os.makedirs('models', exist_ok=True)

print("="*60)
print("BLOOD DONOR PREDICTION SYSTEM")
print("="*60)

# Load data
print("\n1. Loading data...")
data = pd.read_csv('transfusion.csv')
data.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donated']

print(f"   Dataset shape: {data.shape}")
print(f"   Columns: {data.columns.tolist()}")
print(f"   Donors: {data['Donated'].sum()} ({data['Donated'].mean()*100:.2f}%)")

# Feature Engineering
print("\n2. Feature Engineering...")
data['Freq_Monetary'] = data['Frequency'] * data['Monetary']
data['Recency_Freq'] = data['Recency'] * data['Frequency']
data['Donation_Rate'] = data['Frequency'] / (data['Time'] + 1)
data['Monetary_Rate'] = data['Monetary'] / (data['Time'] + 1)

feature_cols = ['Recency', 'Frequency', 'Monetary', 'Time', 
                'Freq_Monetary', 'Recency_Freq', 'Donation_Rate', 'Monetary_Rate']

X = data[feature_cols]
y = data['Donated']

print(f"   Features: {len(feature_cols)}")

# Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

# Define models
print("\n4. Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"   Training {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'model': model
    }
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name]['roc_auc'] = auc
        print(f"Acc: {accuracy:.4f}, AUC: {auc:.4f}")
    else:
        print(f"Acc: {accuracy:.4f}")

# Find best model
print("\n5. Finding best model...")
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"   🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")

# Hyperparameter tuning for best model
print("\n6. Hyperparameter tuning...")
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    xgb_tune = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_tune, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")

# Save best model
print("\n7. Saving models...")
joblib.dump(best_model, 'models/best_model.pkl')
print("   ✅ Models saved to 'models/' directory")

# Visualization
print("\n8. Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison
ax = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
colors = ['#ff6b6b' if m == best_model_name else '#4ecdc4' for m in model_names]
bars = ax.barh(model_names, accuracies, color=colors)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 1.0)
# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, 
            f'{acc:.3f}', ha='right', va='center', fontweight='bold', color='white')

# 2. Confusion Matrix for Best Model
ax = axes[0, 1]
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Won\'t Donate', 'Will Donate'],
            yticklabels=['Won\'t Donate', 'Will Donate'])
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

# 3. Feature Importance
ax = axes[1, 0]
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
    imp_df = imp_df.sort_values('Importance', ascending=True)
    colors_imp = ['#ff6b6b' if i == 0 else '#4ecdc4' for i in range(len(imp_df))]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp)
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)
    # Add value labels
    for i, (_, row) in enumerate(imp_df.iterrows()):
        ax.text(row['Importance'] + 0.01, i, f'{row["Importance"]:.3f}', va='center')

# 4. ROC-AUC Comparison
ax = axes[1, 1]
roc_aucs = [results[m].get('roc_auc', 0) for m in model_names]
bars_roc = ax.barh(model_names, roc_aucs, color='#a8e6cf')
ax.set_xlabel('ROC-AUC Score', fontsize=12)
ax.set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 1.0)
# Add value labels
for bar, auc in zip(bars_roc, roc_aucs):
    if auc > 0:
        ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', ha='right', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ Visualization saved as 'model_analysis.png'")

# Summary Report
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"\n✅ Best Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   Features used: {len(feature_cols)}")
print(f"\n📊 Model Performance:")
print("-"*40)
results_df = pd.DataFrame([
    {'Model': m, 'Accuracy': results[m]['accuracy'], 
     'ROC-AUC': results[m].get('roc_auc', 0)} 
    for m in model_names
]).sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))
print("\n" + "="*60)
print("✅ Analysis Complete! Ready to run web app.")
print("   Run: python app.py")
print("="*60)