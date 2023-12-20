from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model
model_rf = pickle.load(open("randomfores.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl", "rb"))  # Load the scaler

# Render the main page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        age = float(request.form['age'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        gender = int(request.form['gender'])
        ever_married = int(request.form['ever_married'])
        residence_type = int(request.form['Residence_type'])
        work_type = int(request.form['work_type'])
        smoking_status = int(request.form['smoking_status'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],  # Include hypertension column
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        })

        # Select only the columns used during training
        input_data_numeric = input_data[['age', 'avg_glucose_level', 'bmi']]

        # Make prediction
        try:
            print("Input Data before scaling:", input_data)
            input_data_scaled_numeric = scaler.transform(input_data_numeric)
            print("Input Data after scaling:", input_data_scaled_numeric)

            input_data_scaled = pd.DataFrame(input_data_scaled_numeric, columns=['age', 'avg_glucose_level', 'bmi'])
            print("Input Data after scaling (DataFrame):", input_data_scaled)

            # Add back the non-numeric columns
            input_data[['age', 'avg_glucose_level', 'bmi']] = input_data_scaled
            print("Final Input Data for Prediction:", input_data)

            prediction = model_rf.predict(input_data)
            print("Prediction:", prediction)
        except ValueError as e:
            return f"Error: {str(e)}"
        

        if prediction[0] == 0:
            result = "No Stroke"
        else:
            result = "Stroke"

        return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
