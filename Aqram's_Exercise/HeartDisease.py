from flask import Flask, render_template,  redirect, url_for, session
from flask_wtf import FlaskForm 
from wtforms import TextField,SubmitField 
from wtforms.validators import NumberRange 
from joblib import dump, load


import numpy as np  



def return_prediction(model, scaler, sample_json):
    
    age = sample_json['age']
    sex = sample_json['sex']
    cp = sample_json['cp']
    trestbps = sample_json['trestbps']
    chol = sample_json['chol']
    fbs = sample_json['fbs']
    restecg = sample_json['restecg']
    thalach = sample_json['thalach']
    exang = sample_json['exang']
    oldpeak = sample_json['oldpeak']
    slope = sample_json['slope']
    ca = sample_json['ca']
    thal = sample_json['thal']
    
    heart = [[age,sex ,cp ,trestbps ,chol ,fbs ,restecg ,thalach ,exang ,oldpeak ,slope ,ca ,thal]]
    
    heart = scaler.transform(heart)
    
    prediction = model.predict(heart)
    
    return prediction[0]


app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey' 


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
heart_model = load('heart_model.h5')
heart_scaler = load('heart_scaler.pkl')


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class HouseForm(FlaskForm): 
    age = TextField('age')
    sex = TextField('sex')
    cp = TextField('cp')
    trestbps = TextField('trestbps')
    chol = TextField('chol')
    fbs = TextField('fbs')
    restecg = TextField('restecg')
    thalach = TextField('thalach')
    exang = TextField('exang')
    oldpeak = TextField('oldpeak')
    slope = TextField('slope')
    ca = TextField('ca')
    thal = TextField('thal')
   
    submit = SubmitField('Analyze') 

@app.route('/', methods=['GET', 'POST'])
def index():

    form = HouseForm()
    if form.validate_on_submit(): 
        # Grab the data from the breed on the form.

        session['age'] = form.age.data
        session['sex'] = form.sex.data
        session['cp'] = form.cp.data
        session['trestbps'] = form.trestbps.data
        session['chol'] = form.chol.data
        session['fbs'] = form.fbs.data
        session['restecg'] = form.restecg.data
        session['thalach'] = form.thalach.data
        session['exang'] = form.exang.data
        session['oldpeak'] = form.oldpeak.data
        session['slope'] = form.slope.data
        session['ca'] = form.ca.data
        session['thal'] = form.thal.data
        
        return redirect(url_for("prediction"))


    return render_template('HeartDisease_HTML.html', form=form) 


@app.route('/predict')
def prediction():

    content = {}

    content['Lot Frontage'] = float(session['Lot_Frontage']) 
    content['Lot Area'] = float(session['Lot_Area']) 
    content['Overall Qual'] = float(session['Overall_Qual'])
    content['Year Built'] = float(session['Year_Built'])
    content['Year Remod/Add'] = float(session['Year_Remod'])
    content['Exter Qual'] = float(session['Exter_Qual'])
    content['Bsmt Qual'] = float(session['Bsmt_Qual'])
    content['Bsmt Exposure'] = float(session['Bsmt_Exposure'])
    content['BsmtFin SF 1'] = float(session['BsmtFin_SF_1'])
    content['Total Bsmt SF'] = float(session['Total_Bsmt_SF'])
    content['1st Flr SF'] = float(session['First_Flr_SF'])
    content['Gr Liv Area'] = float(session['Gr_Liv_Area'])
    content['Kitchen Qual'] = float(session['Kitchen_Qual'])
    content['Fireplace Qu'] = float(session['Fireplaces'])
    content['Garage Finish'] = float(session['Garage_Finish'])
    content['Garage Cars'] = float(session['Garage_Cars'])
    content['Garage Area'] = float(session['Garage_Area'])
    content['Neighborhood_NridgHt'] = float(session['Neighborhood_NridgHt'])
    content['Sale Condition_Partial'] = float(session['Sale_Condition_Partial'])

    results = return_prediction(model=house_model,scaler=house_scaler,sample_json=content) 

    return render_template('predictions.html',results=round(results,2))


if __name__ == '__main__':
    app.run(debug=True)