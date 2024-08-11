from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved models
dtc_model = joblib.load('decision_tree_model.pkl')
rfc_model = joblib.load('random_forest_model.pkl')

# Reverse mapping
reverse_drug_mapping = {0: 'Drug-B', 1: 'Drug-C', 2: 'Drug-A', 3: 'Drug-X', 4: 'Drug-Y'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bp = int(request.form['bp'])
        cholesterol = int(request.form['cholesterol'])
        na_to_k = float(request.form['na_to_k'])
        model_type = request.form.get('model', 'rfc')  # default to random forest if no model specified

        features = np.array([age, sex, bp, cholesterol, na_to_k]).reshape(1, -1)

        if model_type == 'dtc':
            prediction = dtc_model.predict(features)
        elif model_type == 'rfc':
            prediction = rfc_model.predict(features)
        else:
            return jsonify({'error': 'Invalid model type specified'})

        # Get the drug name from the numeric prediction
        predicted_drug = reverse_drug_mapping[int(prediction[0])]

        return redirect(url_for('result', prediction=predicted_drug))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
