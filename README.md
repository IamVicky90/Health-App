# Health App based on 4 Machine Learning and 2 Deep Learnings at same place
## 6 Different Health Predictions Models I trained
## Table of Contents.
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Directory Tree](#directory-tree)
  * [Libraries](#libraries)
  
  ## Demo
  Deployment Link. https://health-app-by-vicky.azurewebsites.net .
  
 
 __Frontent based on html, css.__
  
  ![error check your internet](https://github.com/IamVicky90/Health-App/blob/main/git%20images/Capture.PNG)
  
  
  __Click the images for the predictions.__
  
  ![error check your internet](https://github.com/IamVicky90/Health-App/blob/main/git%20images/1.PNG)
  
 

  
  ## Overview
I made this six medical prediction applications. For this I use 4 Machine Learning and 1 Deep Learning Models on Flask app trained on the top of Keras API. Just fill the information then it will predict you whether you have some disease or not, provided the data that you have given to it.
## Motivation
What could be a perfect way to utilize unfortunate lockdown period? Like most of you, I spend my time in cooking, Netflix, coding and reading some latest research papers on weekends. This idea came into my mind when I exploring the Kaggle datasets in medical fields. The people now a days paying high fees and give a lot of time to the disease specialists and in testing. So I utilize my best knowledge to overcome their expenses and time. 

## Technical Aspect
This project is divided into two part:
1. Training a deep learning and machine learning model using Keras.
2. Building deployed, and hosting a Flask web app on Microsoft Azure( https://health-app-by-vicky.azurewebsites.net ).

## Installation
The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [downloading it](https://github.com/IamVicky90/Plant-Disease-Prediction/archive/main.zip):
```bash
pip install -r requirements.txt
```
## Run
To run the app in a local machine, shoot this command in the project directory:
__Run the follwing command after installing requirements.txt__
```bash
python app.py
```
## Directory Tree 
```
├── .vscode
│   ├── settings.json
├── Model
│   ├── Chest XRay Pnemonia xception model.h5
│   ├── Heart_disease_ab_0.90_model.sav
│   ├── brest_cancer_rf_model.sav
│   ├── diabetes_xg_0.76_model.sav
│   ├── kidney_disease_ab1_model.sav
│   └── skin cancer vgg16 model.h5
├── chest_xray
│   ├── Test
    │   ├── NORMAL.
    │   ├── PNEUMONIA
│   ├── Train
    │   ├── NORMAL.
    │   ├── PNEUMONIA
│   ├── Val
    │   ├── NORMAL.
    │   ├── PNEUMONIA
├── git images
│   ├── 2.PNG
│   ├── capture.PNG

├── .ipynb notebooks
│   ├── Breast Cancer.ipynb
│   ├── Chest XRay Pnemonia.ipynb
│   ├── Heart Disease.ipynb
│   ├── Kidney Disease.ipynb
│   ├── Real_skin_cancer.ipynb
│   └── diabetes.ipynb

├── skin cancer dl
│   ├── Test
    │   ├── benign
    │   ├── malignant
│   ├── Train
    │   ├── benign
    │   ├── malignant

├── static
│   ├── css
│   ├── Images
│   ├── js

├── templates
│   ├── base.html
│   ├── breast.html
│   ├── diabtes.html
│   ├── heart.html
│   ├── index.html
│   ├── kidney.html
│   ├── pnemonia.html
│   ├── skin.html
│   ├── skin_base.html


├── Breast_cancer_data.csv
├── app.py
├── cancer desc.txt
├── diabetes.csv
├── heart.csv
├── kidney disease desc.txt
├── kidney_disease.csv 
├── README.md
├── requirements.txt
```
## Libraries
also mentioned in [requirements.txt](https://github.com/IamVicky90/Health-App/blob/main/requirements.txt)
```
gunicorn
absl-py==0.11.0
astunparse==1.6.3
cachetools==4.2.1
certifi==2020.12.5
chardet==4.0.0
click==7.1.2
Flask==1.1.2
flatbuffers==1.12
gast==0.3.3
google-auth==1.25.0
google-auth-oauthlib==0.4.2
google-pasta==0.2.0
grpcio==1.32.0
h5py==2.10.0
idna==2.10
itsdangerous==1.1.0
Jinja2==2.11.3
joblib==1.0.0
jsonify==0.5
Keras-Preprocessing==1.1.2
Markdown==3.3.3
MarkupSafe==1.1.1
numpy==1.19.5
oauthlib==3.1.0
opt-einsum==3.3.0
pandas==1.2.1
protobuf==3.14.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
python-dateutil==2.8.1
pytz==2021.1
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7
scikit-learn==0.24.1
scipy==1.6.0
six==1.15.0
sklearn==0.0
tensorboard==2.4.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.4.1
tensorflow-estimator==2.4.0
termcolor==1.1.0
threadpoolctl==2.1.0
typing-extensions==3.7.4.3
urllib3==1.26.3
Werkzeug==1.0.1
wincertstore==0.2
wrapt==1.12.1
xgboost==0.90
```

