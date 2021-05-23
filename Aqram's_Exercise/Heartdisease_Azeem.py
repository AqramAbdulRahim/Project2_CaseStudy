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
    

    heart = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
    heart = scaler.transform(heart)
    
    prediction = model.predict(heart)
    
    if prediction == 0:
        return 'You are at risk of being affected'
    else:
        return 'You are not affected    '


app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey' 


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
heart_model = load("heart_model.h5")
heart_scaler = load("heart_scaler.pkl")


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


    return render_template('HeartDisease_Template.html', form=form) 




@app.route('/predict')
def prediction():

    content = {}

    content['age'] = float(session['age']) 
    content['sex'] = float(session['sex']) 
    content['cp'] = float(session['cp'])
    content['trestbps'] = float(session['trestbps'])
    content['chol'] = float(session['chol'])
    content['fbs'] = float(session['fbs'])
    content['restecg'] = float(session['restecg'])
    content['thalach'] = float(session['thalach'])
    content['slope'] = float(session['slope'])
    content['exang'] = float(session['exang'])
    content['oldpeak'] = float(session['oldpeak'])
    content['ca'] = float(session['ca'])
    content['thal'] = float(session['thal'])





    results = return_prediction(model=heart_model,scaler=heart_scaler,sample_json=content) 


    if content['sex'] == 0:
        gen = 'Female'
    else:
        gen = 'Male'

    if content['cp'] == 1:
        c = 'Typical Angina'
    elif content['cp'] == 2:
        c = 'Atypical Angina'
    elif content['cp'] == 3:
        c = 'Non-anginal Pain'
    else:
        c = 'Asymptomatic'

    if content['fbs'] == 0:
        blood = 'Low'
    else:
        blood = 'High'

    if content['restecg'] == 0:
        ecg = 'Normal'
    elif content['restecg'] == 1:
        ecg = 'ST-T Wave Abnormality'
    else:
        ecg = 'Left Ventricular Hypertrophy'

    if content['exang'] == 0:
        exercise = 'No'
    else:
        exercise = 'Yes'

    if content['slope'] == 1:
        sl = 'Unsloping'
    elif content['slope'] == 2:
        sl = 'Flat'
    else:
        sl = 'Downsloping'

    if content['thal'] == 0:
        th = 'Null'
    elif content['thal'] == 1:
        th = 'Normal'
    elif content['thal'] == 2:
        th = 'Fixed Defects'
    else: 
        th = 'Reversible Defects'

    return render_template('predictions.html',results=(results),gender=(gen),ccp=(c),fblood=(blood),ec=(ecg),ex=(exercise),slo=(sl),tha=(th))


"""
def gender():
        if content['sex'] == 0:
            gen = 'Female'
        else:
            gen = 'Male'

    content['age']=gen
return gen
"""

if __name__ == '__main__':
    app.run(debug=True)