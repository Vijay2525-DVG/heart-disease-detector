// script.js — builds inputs dynamically from feature list and sends predict request


const FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];


const container = document.getElementById('inputs');
const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');


// Build inputs
FEATURE_NAMES.forEach(name => {
const div = document.createElement('div');
div.className = 'field';


const label = document.createElement('label');
label.textContent = name;
label.htmlFor = name;


const input = document.createElement('input');
input.type = 'text';
input.id = name;
input.name = name;
input.placeholder = name;


div.appendChild(label);
div.appendChild(input);
container.appendChild(div);
});


form.addEventListener('submit', async (e) => {
e.preventDefault();
resultDiv.textContent = 'Predicting...';


// Collect values as numbers
const values = FEATURE_NAMES.map(name => {
const val = document.getElementById(name).value;
return val === '' ? null : Number(val);
});


// Simple validation
if (values.some(v => v === null || Number.isNaN(v))) {
resultDiv.textContent = 'Please fill all fields with numeric values.';
return;
}


const payload = { features: values };


try {
const res = await fetch('/predict', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(payload)
});


const data = await res.json();
if (res.ok) {
const p = data.prediction === 1 ? '⚠️ Heart Disease Detected' : '✔️ No Heart Disease';
const prob = data.probability_of_disease !== null ? ` (probability: ${ (data.probability_of_disease*100).toFixed(1)}% )` : '';
resultDiv.textContent = p + prob;
} else {
resultDiv.textContent = 'Error: ' + (data.error || 'Unknown');
}
} catch (err) {
resultDiv.textContent = 'Network error: ' + err.message;
}
});