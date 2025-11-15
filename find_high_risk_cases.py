import pickle
import pandas as pd
import numpy as np

# Load the model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Searching for high-risk cases (probability ≥ 70%)...\n")

# Load original dataset to find actual high-risk patterns
df = pd.read_csv('heart.csv')
df = df.rename(columns={
    'trestbps': 'trtbps',
    'thalach': 'thalachh',
    'exang': 'exng',
    'slope': 'slp',
    'ca': 'caa',
    'thal': 'thall'
})

X = df.drop('target', axis=1)
y = df['target']

# Get predictions for all data
probs = model.predict_proba(X)
disease_probs = probs[:, 1]

# Find cases with disease probability >= 70%
high_risk_indices = np.where(disease_probs >= 0.70)[0]

print(f"Found {len(high_risk_indices)} cases with ≥70% disease probability\n")
print("="*80)
print("TOP 10 HIGHEST RISK CASES FROM DATASET:")
print("="*80)

# Sort by probability and show top 10
sorted_indices = np.argsort(disease_probs)[::-1][:10]

for i, idx in enumerate(sorted_indices, 1):
    prob = disease_probs[idx]
    actual = y.iloc[idx]
    
    print(f"\n#{i} - Disease Probability: {prob:.1%} (Actual: {'Disease' if actual == 1 else 'No Disease'})")
    print(f"   Age: {int(X.iloc[idx]['age'])}")
    print(f"   Sex: {int(X.iloc[idx]['sex'])}")
    print(f"   CP: {int(X.iloc[idx]['cp'])}")
    print(f"   Resting BP: {int(X.iloc[idx]['trtbps'])}")
    print(f"   Cholesterol: {int(X.iloc[idx]['chol'])}")
    print(f"   FBS: {int(X.iloc[idx]['fbs'])}")
    print(f"   Rest ECG: {int(X.iloc[idx]['restecg'])}")
    print(f"   Max HR: {int(X.iloc[idx]['thalachh'])}")
    print(f"   Exercise Angina: {int(X.iloc[idx]['exng'])}")
    print(f"   Oldpeak: {X.iloc[idx]['oldpeak']}")
    print(f"   Slope: {int(X.iloc[idx]['slp'])}")
    print(f"   CA: {int(X.iloc[idx]['caa'])}")
    print(f"   Thal: {int(X.iloc[idx]['thall'])}")

# Create some synthetic high-risk test cases
print("\n" + "="*80)
print("SYNTHETIC HIGH-RISK TEST CASES:")
print("="*80)

test_cases = [
    {
        'name': 'High Risk Case 1 - Multiple risk factors',
        'age': 62, 'sex': 1, 'cp': 0, 'trtbps': 160, 'chol': 290,
        'fbs': 1, 'restecg': 2, 'thalachh': 120, 'exng': 1,
        'oldpeak': 3.5, 'slp': 2, 'caa': 3, 'thall': 3
    },
    {
        'name': 'High Risk Case 2 - Elderly with symptoms',
        'age': 70, 'sex': 1, 'cp': 0, 'trtbps': 165, 'chol': 280,
        'fbs': 1, 'restecg': 2, 'thalachh': 110, 'exng': 1,
        'oldpeak': 4.0, 'slp': 2, 'caa': 3, 'thall': 3
    },
    {
        'name': 'High Risk Case 3 - Low heart rate + high vessels',
        'age': 58, 'sex': 1, 'cp': 0, 'trtbps': 150, 'chol': 270,
        'fbs': 0, 'restecg': 1, 'thalachh': 105, 'exng': 1,
        'oldpeak': 3.0, 'slp': 2, 'caa': 4, 'thall': 3
    },
    {
        'name': 'High Risk Case 4 - Severe ST depression',
        'age': 65, 'sex': 1, 'cp': 0, 'trtbps': 170, 'chol': 300,
        'fbs': 1, 'restecg': 2, 'thalachh': 115, 'exng': 1,
        'oldpeak': 5.0, 'slp': 2, 'caa': 3, 'thall': 3
    }
]

for test in test_cases:
    name = test.pop('name')
    test_df = pd.DataFrame([test])
    pred = model.predict(test_df)[0]
    prob = model.predict_proba(test_df)[0]
    
    print(f"\n{name}:")
    print(f"   Disease Probability: {prob[1]:.1%}")
    print(f"   Status: {'✓ RED' if prob[1] >= 0.70 else '✗ Not high enough'}")
    if prob[1] >= 0.70:
        print(f"\n   USE THESE VALUES IN YOUR UI:")
        for key, val in test.items():
            print(f"   {key}: {val}")

print("\n" + "="*80)
print("Copy the values from any case marked '✓ RED' to test in your UI!")
print("="*80)