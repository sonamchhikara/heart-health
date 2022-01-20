from flask import Flask, session, request, redirect, url_for, render_template

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

@app.route('/api', methods = ['GET', 'POST'])
def api():
  param1 = request.form['param1']
  param2 = request.form['param2']
  param3 = request.form['param3']
  param4 = request.form['param4']
  param5 = request.form['param5']
  return analyse(param1, param2, param3, param4, param5);  

def analyse(param1, param2, param3, param4, param5):
  return "YOUR_CODE_OUTPUT";

# run the application
if __name__ == "__main__":
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(debug=True)