# https://docs.python.org/3/
# https://flask.palletsprojects.com/en/3.0.x/
# https://developer.mozilla.org/en-US/docs/Web/HTML
# https://developer.mozilla.org/en-US/docs/Web/CSS
# https://github.com/MohamedElweza/Multiple-Disease-Prediction-System-using-Machine-Learning/blob/main/Multiple%20Disease%20Prediction(final).py
# https://github.com/Ajay-singhh/Multiple-Disease-Prediction-Webapp-With-MachineLearning/blob/main/predictive.py


from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('Kidney.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        specific_gravity = float(request.form['specific_gravity'])
        albumin = float(request.form['albumin'])
        red_blood_cells = float(request.form['red_blood_cells'])
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        sodium = float(request.form['sodium'])
        haemoglobin = float(request.form['haemoglobin'])
        packed_cell_volume = float(request.form['packed_cell_volume'])
        white_blood_cell = float(request.form['white_blood_cell'])
        hypertension = float(request.form['hypertension'])
        diabetes_mellitus = float(request.form['diabetes_mellitus'])
        coronary_artery_disease = float(request.form['coronary_artery_disease'])
        

        values = np.array([[age, blood_pressure, specific_gravity, albumin, red_blood_cells, blood_glucose_random, blood_urea, serum_creatinine, sodium, haemoglobin, packed_cell_volume, white_blood_cell, hypertension, diabetes_mellitus, coronary_artery_disease]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

