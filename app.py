#ML LIB
import numpy as np
import pickle

# Flask LIB
from flask import Flask,request, render_template

app = Flask(__name__)

model = pickle.load(open('models/predictSalary.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('projectsMl.html')

@app.route('/predict/output', methods=['POST'])
def predictOutput():
    '''
    For rendinering the putput in the web
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('projectsMl.html', prediction_text='Employee Salary should be Rs. {}'.format(output))

@app.route('/test')
def test():
    return render_template('ache.html')


if __name__ == "__main__":
    app.run(debug=True)
