#Import necessary libraries
from flask import Flask,request,render_template
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
import os
import xgboost
from werkzeug.utils import secure_filename
# Create flask instance
app = Flask(__name__)
def pred_pnemoian(img_path):
    model=load_model(r"Model\Chest XRay Pnemonia xception model.h5")
    img=load_img(img_path,target_size=(224,224))
    x=img_to_array(img)/225
    x=np.expand_dims(x, axis=0)
    pred=model.predict(x)
    output=np.argmax(pred,axis=1)
    print(output)
    if output==0:
        return "Prediction Result: Don't worry You don't have any disease!"
    elif output==1:
        return "We found that you have Pnemonia disease please consult with the doctor"
def pred_skin(img_path):
    model=load_model(r"Model\skin cancer vgg16 model.h5")
    img=load_img(img_path,target_size=(224,224))
    x=img_to_array(img)/225
    x=np.expand_dims(x, axis=0)
    pred=model.predict(x)
    output=np.argmax(pred,axis=1)
    print(output)
    if output==0:
        return "Prediction Result: Don't worry You don't have any disease!"
    elif output==1:
        return "We found that you have skin cancer, please consult with the doctor"


# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')
@app.route("/heart", methods=['GET', 'POST'])
def heart():
    return render_template('heart.html')

@app.route('/heart_predict',methods=["GET","POST"])
def heart_predict():
    model = pickle.load(open('Model\Heart_disease_ab_0.90_model.sav', 'rb'))
    print("@@ Heart Disease Model Loaded")
    if request.method == 'POST':
        age=request.form['age']
        
        sex=request.form['sex']
        cp=request.form['cp']
        trestbps=request.form['trestbps']
        chol=request.form['chol']
        fbs=request.form['fbs']
        restecg=request.form['restecg']
        thalach=request.form['thalach']
        exang=request.form['exang']
        oldpeak=request.form['oldpeak']
        slope=request.form['slope']

        ca=request.form['ca']

        thal=request.form['thal']
        values=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        print(values)
        X=[]
        try:
            for value in values:
                X.append(np.log(float(value)+1))
            output=model.predict([X])
            print(output)
        except Exception as e:
            print("@@",e)
            return render_template('heart.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")

        if output==0:
            return render_template('heart.html',prediction_text="Prediction Result: Don't worry You don't have any disease!")
        elif output==1:
            return render_template('heart.html',prediction_text="We found something wrong with you please consult with the doctor")




    else:
        return render_template('heart.html')

@app.route("/breast", methods=['GET', 'POST'])
def breast():
    return render_template('breast.html')
@app.route("/breast_predict", methods=['GET', 'POST'])
def breast_predict():
    model = pickle.load(open(r'Model\brest_cancer_rf_model.sav', 'rb'))
    print("@@ Breast Cancer Model Loaded")
    if request.method == 'POST':
        try:
            mean_radius=float(request.form['mean_radius'])
            mean_texture=float(request.form['mean_texture'])
            mean_perimeter=float(request.form['mean_perimeter'])
            mean_area=float(request.form['mean_area'])
            mean_smoothness=float(request.form['mean_smoothness'])
        except Exception as e:
            print("@@",e)
            return render_template('breast.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")
        
        output=model.predict([[mean_radius,mean_texture,mean_perimeter,mean_smoothness]])
        print(output)
        if output==0:
            return render_template('breast.html',prediction_text="Prediction Result: Don't worry You don't have any disease!")
        elif output==1:
            return render_template('breast.html',prediction_text="We found something wrong with you please consult with the doctor")

    return render_template('breast.html')
@app.route("/pnemonia", methods=['GET', 'POST'])
def pnemonia():
    return render_template('pnemonia.html')
@app.route("/predict_pnemonia", methods=['GET', 'POST'])
def predict_pnemonia():
    if request.method=='POST':
        f = request.files['file']
        print("inside predict_pnemonia")

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = pred_skin(file_path)
        result=preds
        return result
    
@app.route("/diabtes", methods=['GET', 'POST'])
def diabtes():
    return render_template('diabtes.html')
@app.route("/diabtes_predict", methods=['GET', 'POST'])
def diabtes_predict():
    model = pickle.load(open(r'Model\diabetes_xg_0.76_model.sav', 'rb'))
    print("@@ Diabtes Model Loaded")
    if request.method == 'POST':
        try:
            Pregnancies=float(request.form['Pregnancies'])
            Glucose=float(request.form['Glucose'])
            BloodPressure=float(request.form['BloodPressure'])
            SkinThickness=float(request.form['SkinThickness'])
            Insulin=float(request.form['Insulin'])
            BMI=float(request.form['BMI'])
            DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
            Age=float(request.form['Age'])
        except Exception as e:
            print("@@",e)
            return render_template('diabtes.html',prediction_text="Some unknown error occured please input the values in number or contact the develpor if it still occurs")
        df=pd.DataFrame([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        output=model.predict(df)
        print(output)
        if output==0:
            return render_template('diabtes.html',prediction_text="Prediction Result: Don't worry You don't have diabtes!")
        elif output==1:
            return render_template('diabtes.html',prediction_text="We found that you have diabtes, please consult with the doctor")

    return render_template('diabtes.html')
@app.route("/skin", methods=['GET', 'POST'])
def skin():
    return render_template('skin.html')
@app.route("/predict_skin", methods=['GET', 'POST'])
def predict_skin():
    if request.method=='POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = pred_pnemoian(file_path)
        result=preds
        return result
@app.route("/kidney", methods=['GET', 'POST'])
def kidney():
    return render_template('kidney.html')



if __name__ == "__main__":
    app.run(threaded=False)