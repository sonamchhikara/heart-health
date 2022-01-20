from flask import Flask, request, render_template
import pandas as pd
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
	return render_template('home.html')

def getParameters():
    parameters = []
    #parameters.append(request.form('name'))
    parameters.append(request.form['age'])
    parameters.append(request.form['sex'])
    parameters.append(request.form['chest_pain_type'])
    parameters.append(request.form['resting_blood_pressure'])
    parameters.append(request.form['cholesterol'])
    parameters.append(request.form['fasting_blood_sugar'])
    parameters.append(request.form['rest_ecg'])
    parameters.append(request.form['max_heart_rate_achieved'])
    parameters.append(request.form['exercise_induced_angina'])
    parameters.append(request.form['st_depression'])
    parameters.append(request.form['st_slope'])
    parameters.append(request.form['target'])
    return parameters

@app.route('/predict',methods=['POST'])
def predict():
    model = open("model.pkl","rb")
    clfr = joblib.load(model)

    if request.method == 'POST':
        parameters = getParameters()
        inputFeature = np.asarray(parameters).reshape(1,-1)
        my_prediction = clfr.predict(inputFeature)
    return render_template('result.html',prediction = int(my_prediction[0]))

if __name__ == '__main__':
	app.run(debug=True)
