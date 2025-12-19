import os
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Get absolute path to model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'model', 'random_forest_regressor_model.pkl')

# Load model safely
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Study_Hours_per_Week = float(request.form['Study_Hours_per_Week'])
        Attendance_Rate = float(request.form['Attendance_Rate'])
        Past_Exam_Scores = float(request.form['Past_Exam_Scores'])
        Extracurricular_Activities = request.form['Extracurricular_Activities']

        # One-hot encode categorical feature
        if Extracurricular_Activities.lower() == 'yes':
            Extra_No, Extra_Yes = 0, 1
        else:
            Extra_No, Extra_Yes = 1, 0

        # Prepare input (match model’s feature count)
        final_features = np.array([[Study_Hours_per_Week, Attendance_Rate,
                                    Past_Exam_Scores, Extra_No, Extra_Yes]])
        prediction = model.predict(final_features)[0]

        return render_template('index.html',
                               prediction_text=f'🎯 Predicted Final Exam Score: {prediction:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'⚠️ Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
