from flask import Flask, request, render_template
import pandas as pd
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

def index():
   if 'username' in session:
      username = session['username']
         return 'Logged in as ' + username + '<br>' + "<b><a href = '/logout'>click here to log out</a></b>

if __name__ == '__main__':
	app.run(debug=True)
