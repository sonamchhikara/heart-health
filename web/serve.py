from flask import Flask, session, request, redirect, url_for, render_template
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random

# creates a Flask application, named app
app = Flask(__name__)

@app.route('/')
def index():
  if 'email' in session:
    email = session['email']
    name = "test"
    return render_template('home.html', email=email, firstName=name)
  return render_template('login.html')

@app.route('/login', methods = ['GET', 'POST'])
def login():
  if request.method == 'POST':
    if request.form['email'] == 'test@test.com' and request.form['password'] == 'test':
      session['email'] = request.form['email']
      session['password'] = request.form['password']
      return redirect(url_for('index'))
    return render_template('login.html')
  if 'email' in session:
    return redirect(url_for('index'))
  return render_template('login.html')

@app.route('/logout', methods = ['GET', 'POST'])
def logout():
  session.pop('email', None)
  return render_template('login.html')

@app.route('/api', methods = ['GET'])
def api():
  
  param1 = request.args.get('bp')
  param2 = request.args.get('cholestrol')
  param3 = request.args.get('sugar_level')
  param4 = request.args.get('heart_rate')
  param5 = request.args.get('smoking')
  print (request.args)

  
  return analyse(param1, param2, param3, param4, param5)

def analyse(param1, param2, param3, param4, param5):
  print("ok")
  parameters = []
  parameters.append(int(param1))
  parameters.append(int(param2))
  parameters.append(int(param3))
  parameters.append(int(param4))


  model = open("model_j_xgb_5_new.pkl","rb")
  clfr = joblib.load(model)
  simple_list = [parameters]
  df = pd.DataFrame(simple_list, columns = ['resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar','max_heart_rate_achieved'])
  scaler = MinMaxScaler()
  df[['resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar','max_heart_rate_achieved']] = scaler.fit_transform(df[['resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar','max_heart_rate_achieved']])

  print(df)
  my_prediction = clfr.predict(df)
  print(my_prediction[0])
  if(my_prediction[0]==1):
    result = random.randint(100, 170)
  else:
    result = random.randint(80, 100)

  return str(result)

  #return "YOUR_CODE_OUTPUT";

# run the application
if __name__ == "__main__":
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(debug=True)