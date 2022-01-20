from flask import Flask, session, request, redirect, url_for, render_template
import joblib
import numpy as np
# creates a Flask application, named app
app = Flask(__name__, template_folder="templates")

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
  #print (request.args.get)
  #temp = request.args
  return analyse(param1, param2, param3, param4, param5)

def analyse(param1, param2, param3, param4, param5):
  print("ok")
  parameters = []
  parameters.append(int(param1))
  parameters.append(int(param2))
  parameters.append(int(param3))
  parameters.append(int(param4))
  parameters.append(int(param5))
  print (parameters)

  model = open("model.pkl","rb")
  clfr = joblib.load(model)

  inputFeature = np.asarray(parameters).reshape(1,-1)
  my_prediction = clfr.predict(inputFeature)

  print(int(my_prediction[0]))

  return (40)#(int(my_prediction[0]))

  #return "YOUR_CODE_OUTPUT";

# run the application
if __name__ == "__main__":
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(debug=True)