#ML LIB
import numpy as np
import pandas as pd
import pickle

# Flask LIB
from flask import Flask,request, render_template
from flask_wtf.csrf import CSRFProtect
from flask_wtf.csrf import CSRFError



app = Flask(__name__)
app.secret_key = 'ERFASDFG@#jnlLKLN'
csrf = CSRFProtect(app)
csrf.init_app(app)

modelSalary = pickle.load(open('models/predictSalary.pkl','rb'))
modelTweet = pickle.load(open('models/RealOrfake.pkl','rb'))
OneHotLoan = pickle.load(open('models/loan/OneHotEncode.pkl','rb'))
modelLoans = pickle.load(open('models/loan/LoanPrediction.pkl','rb'))

def clean_data(name):
    processed = name.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
    processed = processed.str.replace(r'£|\$', 'moneysymb')
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    processed = processed.str.replace(r'\s+', ' ')
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    processed = processed.str.lower()
    return processed


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/salary')
def salary():
    return render_template('SalaryPrediction.html',prediction_text="Essentially, All Models are wrong but some are useful - George Box")

@app.route('/salary/output', methods=['POST'])
def predictSalary():
    '''
    For rendinering the putput in the web
    '''
    csrf_value = request.form["csrf_token"]
    int_features = [int(x) for x in list(request.form.values())[:-1]]
    final_features = [np.array(int_features)]
    prediction = modelSalary.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('SalaryPrediction.html', prediction_text='Employee Salary should be Rs. {}'.format(output))


@app.route('/tweet')
def tweet():
    return render_template('tweetsFR.html',prediction_text="Essentially, All Models are wrong but some are useful - George Box")

@app.route('/tweet/output', methods=['POST'])
def predictTweet():
    csrf_value = request.form["csrf_token"]
    int_features = request.form["tweet"]
    series = pd.Series(int_features)
    data = clean_data(series)
    value = modelTweet.predict(data)
    if value == 1:
        outputs="Real tweet"
    else:
        outputs="Fake tweet"
    return render_template('tweetsFR.html', prediction_text=outputs)


@app.route('/loan')
def loan():
    return render_template('LoanPred.html',prediction_text="Essentially, All Models are wrong but some are useful - George Box")

@app.route('/loan/output', methods=['POST'])
def predictLoan():
    csrf_value = request.form["csrf_token"]
    Loan_Amount_Term = int(request.form["Loan_Amount_Term"])
    LoanAmount = int(request.form["LoanAmount"])
    CoapplicantIncome = int(request.form["CoapplicantIncome"])
    ApplicantIncome = int(request.form["ApplicantIncome"])
    Gender = request.form["Gender"]
    Married = request.form["Married"]
    Dependents = request.form["Dependents"]
    Property_Area = request.form["Property_Area"]
    Education  = request.form["Education"]
    Credit_History = request.form["Credit_History"]
    Self_Employed  = request.form["Self_Employed"]
    arr = OneHotLoan.transform([[Loan_Amount_Term,LoanAmount,CoapplicantIncome,ApplicantIncome,Gender,Married,Dependents,Property_Area,Education,Credit_History,Self_Employed]])
    value = modelLoans['LogisticRegression'].predict(arr)
    if value == 1:
        outputs="You will get a loan"
    else:
        outputs="Sorry,for Loan try again later"
    return render_template('LoanPred.html', prediction_text=outputs)

@app.route('/test')
def test():
    return render_template('ache.html')


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return render_template('csrf_error.html', reason=e.description), 400

if __name__ == "__main__":
    app.run(debug=True)
