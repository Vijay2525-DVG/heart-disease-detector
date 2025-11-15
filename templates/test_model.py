import pickle
import pandas as pd
import numpy as np

# Load the model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print(f"Expected features: {model.feature_names_in_}")

# Test Case 1: Healthy 25-year-old
print("\n" + "="*60)
print("TEST CASE 1: Healthy 25-year-old Female")
print("="*60)

healthy_patient = pd.DataFrame([{
    'age': 25,
    'sex': 0,
    'cp': 0,
    'trtbps': 110,
    'chol': 180,
    'fbs': 0,
    'restecg': 0,
    'thalachh': 180,
    'exng': 0,
    'oldpeak': 0.0,
    'slp': 0,
    'caa': 0,
    'thall': 1
}])

pred = model.predict(healthy_patient)[0]
prob = model.predict_proba(healthy_patient)[0]

print(f"Prediction: {pred} (0=No Disease, 1=Disease)")
print(f"Probability - No Disease: {prob[0]:.3f} ({prob[0]*100:.1f}%)")
print(f"Probability - Disease: {prob[1]:.3f} ({prob[1]*100:.1f}%)")
print(f"Expected: Low Risk / No Disease")
print(f"Result: {'✓ CORRECT' if pred == 0 else '✗ WRONG - Model predicting disease!'}")

# Test Case 2: High Risk 70-year-old
print("\n" + "="*60)
print("TEST CASE 2: High Risk 70-year-old Male")
print("="*60)

high_risk_patient = pd.DataFrame([{
    'age': 70,
    'sex': 1,
    'cp': 3,
    'trtbps': 180,
    'chol': 300,
    'fbs': 1,
    'restecg': 2,
    'thalachh': 100,
    'exng': 1,
    'oldpeak': 5.0,
    'slp': 2,
    'caa': 4,
    'thall': 3
}])

pred2 = model.predict(high_risk_patient)[0]
prob2 = model.predict_proba(high_risk_patient)[0]

print(f"Prediction: {pred2} (0=No Disease, 1=Disease)")
print(f"Probability - No Disease: {prob2[0]:.3f} ({prob2[0]*100:.1f}%)")
print(f"Probability - Disease: {prob2[1]:.3f} ({prob2[1]*100:.1f}%)")
print(f"Expected: High Risk / Disease")
print(f"Result: {'✓ CORRECT' if pred2 == 1 else '✗ WRONG - Model not detecting disease!'}")

# Test Case 3: Middle-aged moderate risk
print("\n" + "="*60)
print("TEST CASE 3: 50-year-old Male with some risk factors")
print("="*60)

moderate_patient = pd.DataFrame([{
    'age': 50,
    'sex': 1,
    'cp': 2,
    'trtbps': 140,
    'chol': 240,
    'fbs': 0,
    'restecg': 1,
    'thalachh': 150,
    'exng': 0,
    'oldpeak': 1.5,
    'slp': 1,
    'caa': 1,
    'thall': 2
}])

pred3 = model.predict(moderate_patient)[0]
prob3 = model.predict_proba(moderate_patient)[0]

print(f"Prediction: {pred3} (0=No Disease, 1=Disease)")
print(f"Probability - No Disease: {prob3[0]:.3f} ({prob3[0]*100:.1f}%)")
print(f"Probability - Disease: {prob3[1]:.3f} ({prob3[1]*100:.1f}%)")

# Check some actual training data
print("\n" + "="*60)
print("CHECKING TRAINING DATA DISTRIBUTION")
print("="*60)

df = pd.read_csv('heart.csv')
print(f"\nYoung patients (age < 35) in dataset: {len(df[df['age'] < 35])}")
print(f"Disease rate in young patients: {df[df['age'] < 35]['target'].mean():.2%}")

print(f"\nOld patients (age > 65) in dataset: {len(df[df['age'] > 65])}")
print(f"Disease rate in old patients: {df[df['age'] > 65]['target'].mean():.2%}")

print(f"\nOverall disease rate: {df['target'].mean():.2%}")

# Check if model is overfitting or has issues
print("\n" + "="*60)
print("MODEL DIAGNOSTICS")
print("="*60)
print(f"Number of trees: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
print(f"Features used: {len(model.feature_names_in_)}")

# Test with actual data from the dataset
print("\n" + "="*60)
print("TESTING WITH ACTUAL DATASET SAMPLES")
print("="*60)

# Rename columns to match model
df_renamed = df.rename(columns={
    'trestbps': 'trtbps',
    'thalach': 'thalachh',
    'exang': 'exng',
    'slope': 'slp',
    'ca': 'caa',
    'thal': 'thall'
})

X = df_renamed.drop('target', axis=1)
y = df_renamed['target']

# Test on 5 random healthy patients (target=0)
healthy_samples = df_renamed[df_renamed['target'] == 0].sample(5)
X_healthy = healthy_samples.drop('target', axis=1)
y_healthy = healthy_samples['target']

predictions = model.predict(X_healthy)
probabilities = model.predict_proba(X_healthy)

print("\n5 Random Healthy Patients from Dataset:")
for i in range(5):
    print(f"  Patient {i+1}: Age={X_healthy.iloc[i]['age']}, Predicted={predictions[i]}, Disease Prob={probabilities[i][1]:.3f}")

accuracy_healthy = (predictions == y_healthy).mean()
print(f"\nAccuracy on healthy samples: {accuracy_healthy:.2%}")