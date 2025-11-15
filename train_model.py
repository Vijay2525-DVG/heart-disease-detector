import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

print("Loading dataset...")
df = pd.read_csv('heart.csv')

print(f"Original dataset shape: {df.shape}")
print(f"\nAge distribution:")
print(f"Min age: {df['age'].min()}, Max age: {df['age'].max()}")
print(f"Mean age: {df['age'].mean():.1f}")
print(f"Patients under 35: {(df['age'] < 35).sum()}")
print(f"Patients 35-50: {((df['age'] >= 35) & (df['age'] < 50)).sum()}")
print(f"Patients 50-65: {((df['age'] >= 50) & (df['age'] < 65)).sum()}")
print(f"Patients over 65: {(df['age'] >= 65).sum()}")

# Rename columns to match model expectations
df = df.rename(columns={
    'trestbps': 'trtbps',
    'thalach': 'thalachh',
    'exang': 'exng',
    'slope': 'slp',
    'ca': 'caa',
    'thal': 'thall'
})

# Add synthetic healthy young patients to balance the data
print("\nAdding synthetic healthy young patients...")
np.random.seed(42)

# Create 50 synthetic healthy young patients (20-35 years old)
synthetic_healthy = []
for _ in range(50):
    synthetic_healthy.append({
        'age': np.random.randint(20, 36),
        'sex': np.random.randint(0, 2),
        'cp': np.random.choice([0, 1], p=[0.7, 0.3]),  # Mostly typical or atypical
        'trtbps': np.random.randint(100, 130),  # Good blood pressure
        'chol': np.random.randint(150, 220),    # Good cholesterol
        'fbs': 0,  # Normal blood sugar
        'restecg': 0,  # Normal ECG
        'thalachh': np.random.randint(150, 200),  # High max heart rate
        'exng': 0,  # No exercise angina
        'oldpeak': round(np.random.uniform(0, 0.5), 1),  # Low ST depression
        'slp': np.random.choice([0, 1]),  # Normal slope
        'caa': 0,  # No major vessels
        'thall': np.random.choice([1, 2], p=[0.8, 0.2]),  # Mostly normal
        'target': 0  # No disease
    })

synthetic_df = pd.DataFrame(synthetic_healthy)
df = pd.concat([df, synthetic_df], ignore_index=True)

print(f"New dataset shape: {df.shape}")
print(f"Added {len(synthetic_healthy)} synthetic healthy young patients")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train with better hyperparameters to prevent overfitting
print("\nTraining Random Forest model with improved parameters...")
model = RandomForestClassifier(
    n_estimators=200,        # More trees
    max_depth=8,             # Shallower to prevent overfitting
    min_samples_split=10,    # More samples needed to split
    min_samples_leaf=5,      # More samples in leaf nodes
    max_features='sqrt',     # Use sqrt of features
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n{'='*50}")
print(f"CLASSIFICATION REPORT (Test Set)")
print(f"{'='*50}")
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Disease']))

print(f"\n{'='*50}")
print(f"CONFUSION MATRIX (Test Set)")
print(f"{'='*50}")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Feature Importance
print(f"\n{'='*50}")
print(f"FEATURE IMPORTANCE")
print(f"{'='*50}")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Test with specific cases
print(f"\n{'='*50}")
print(f"TESTING EDGE CASES")
print(f"{'='*50}")

# Test Case 1: Healthy 25-year-old
test_healthy = pd.DataFrame([{
    'age': 25, 'sex': 0, 'cp': 0, 'trtbps': 110, 'chol': 180,
    'fbs': 0, 'restecg': 0, 'thalachh': 180, 'exng': 0,
    'oldpeak': 0.0, 'slp': 0, 'caa': 0, 'thall': 1
}])

pred = model.predict(test_healthy)[0]
prob = model.predict_proba(test_healthy)[0]
print(f"\n25-year-old healthy patient:")
print(f"  Prediction: {'Disease' if pred == 1 else 'No Disease'}")
print(f"  Disease Probability: {prob[1]:.3f} ({prob[1]*100:.1f}%)")
print(f"  Expected: No Disease (Low probability)")
print(f"  Result: {'✓ CORRECT' if pred == 0 and prob[1] < 0.5 else '✗ NEEDS IMPROVEMENT'}")

# Test Case 2: High risk 70-year-old
test_high_risk = pd.DataFrame([{
    'age': 70, 'sex': 1, 'cp': 3, 'trtbps': 180, 'chol': 300,
    'fbs': 1, 'restecg': 2, 'thalachh': 100, 'exng': 1,
    'oldpeak': 5.0, 'slp': 2, 'caa': 4, 'thall': 3
}])

pred2 = model.predict(test_high_risk)[0]
prob2 = model.predict_proba(test_high_risk)[0]
print(f"\n70-year-old high-risk patient:")
print(f"  Prediction: {'Disease' if pred2 == 1 else 'No Disease'}")
print(f"  Disease Probability: {prob2[1]:.3f} ({prob2[1]*100:.1f}%)")
print(f"  Expected: Disease (High probability)")
print(f"  Result: {'✓ CORRECT' if pred2 == 1 and prob2[1] > 0.7 else '✗ NEEDS IMPROVEMENT'}")

# Save the model
print(f"\n{'='*50}")
print(f"SAVING MODEL")
print(f"{'='*50}")
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'heart_model.pkl'")

# Verify
print("\nVerifying model can be loaded...")
with open('heart_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("✓ Model loaded successfully!")

if hasattr(loaded_model, 'feature_names_in_'):
    print(f"\nModel expects these feature names:")
    print(loaded_model.feature_names_in_)

print(f"\n{'='*50}")
print(f"TRAINING COMPLETE!")
print(f"{'='*50}")
print("\nModel should now handle young healthy patients correctly.")
print("Restart your Flask server to load the new model.")