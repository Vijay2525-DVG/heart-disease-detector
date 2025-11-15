import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

print("Loading dataset...")
df = pd.read_csv('heart.csv')

print(f"Dataset shape: {df.shape}")

# Rename columns
df = df.rename(columns={
    'trestbps': 'trtbps',
    'thalach': 'thalachh',
    'exang': 'exng',
    'slope': 'slp',
    'ca': 'caa',
    'thal': 'thall'
})

# Add just 20 synthetic healthy young patients - minimal augmentation
print("\nAdding minimal synthetic data for young patients...")
np.random.seed(42)

synthetic_healthy = []
for _ in range(20):
    synthetic_healthy.append({
        'age': np.random.randint(22, 35),
        'sex': np.random.randint(0, 2),
        'cp': np.random.choice([0, 1]),
        'trtbps': np.random.randint(105, 125),
        'chol': np.random.randint(160, 210),
        'fbs': 0,
        'restecg': 0,
        'thalachh': np.random.randint(160, 195),
        'exng': 0,
        'oldpeak': round(np.random.uniform(0, 0.3), 1),
        'slp': 0,
        'caa': 0,
        'thall': 1,
        'target': 0
    })

synthetic_df = pd.DataFrame(synthetic_healthy)
df = pd.concat([df, synthetic_df], ignore_index=True)

print(f"New dataset shape: {df.shape}")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nTarget distribution:\n{y.value_counts()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

# Train with optimal parameters
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,           # Good depth for pattern recognition
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\n{'='*50}")
print(f"CLASSIFICATION REPORT")
print(f"{'='*50}")
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Disease']))

print(f"\n{'='*50}")
print(f"CONFUSION MATRIX")
print(f"{'='*50}")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*50}")
print(f"FEATURE IMPORTANCE")
print(f"{'='*50}")
print(feature_importance)

# Test edge cases
print(f"\n{'='*50}")
print(f"TESTING SPECIFIC CASES")
print(f"{'='*50}")

test_cases = [
    {
        'name': '25-year-old healthy',
        'data': {'age': 25, 'sex': 0, 'cp': 0, 'trtbps': 110, 'chol': 180,
                'fbs': 0, 'restecg': 0, 'thalachh': 180, 'exng': 0,
                'oldpeak': 0.0, 'slp': 0, 'caa': 0, 'thall': 1},
        'expected': 'Low Risk'
    },
    {
        'name': '55-year-old with risk factors',
        'data': {'age': 55, 'sex': 1, 'cp': 2, 'trtbps': 145, 'chol': 250,
                'fbs': 1, 'restecg': 1, 'thalachh': 135, 'exng': 1,
                'oldpeak': 2.0, 'slp': 1, 'caa': 2, 'thall': 2},
        'expected': 'High Risk'
    },
    {
        'name': '60-year-old moderate risk',
        'data': {'age': 60, 'sex': 1, 'cp': 3, 'trtbps': 150, 'chol': 240,
                'fbs': 0, 'restecg': 0, 'thalachh': 140, 'exng': 0,
                'oldpeak': 1.5, 'slp': 1, 'caa': 1, 'thall': 2},
        'expected': 'Moderate-High Risk'
    },
    {
        'name': '45-year-old healthy',
        'data': {'age': 45, 'sex': 0, 'cp': 1, 'trtbps': 120, 'chol': 200,
                'fbs': 0, 'restecg': 0, 'thalachh': 170, 'exng': 0,
                'oldpeak': 0.5, 'slp': 0, 'caa': 0, 'thall': 1},
        'expected': 'Low-Moderate Risk'
    }
]

for test in test_cases:
    test_df = pd.DataFrame([test['data']])
    pred = model.predict(test_df)[0]
    prob = model.predict_proba(test_df)[0]
    
    print(f"\n{test['name']}:")
    print(f"  Prediction: {'Disease' if pred == 1 else 'No Disease'}")
    print(f"  Disease Probability: {prob[1]:.3f} ({prob[1]*100:.1f}%)")
    print(f"  Expected: {test['expected']}")

# Save model
print(f"\n{'='*50}")
print(f"SAVING MODEL")
print(f"{'='*50}")
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'heart_model.pkl'")

print("\n✓ Training complete! Restart Flask to use the new model.")