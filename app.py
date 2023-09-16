

import os

import numpy as np

# Keras
import pickle
from tensorflow.keras.models import load_model

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
import librosa
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = 'files'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER1 = 'static/uploads/'

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())
    return sensitivity



model_path2 = 'models/modelV2.h5' # load .h5 Model

custom_objects = {
    'f1_score': f1_score,
    'recall_m': recall_score,
    'precision_m': precision_score,
    'specificity_m': specificity_m,
    'sensitivity_m': sensitivity_m
}


model = load_model(model_path2, custom_objects=custom_objects)

model_name = open("tk.pkl","rb")
scaler = pickle.load(model_name)


def extract_features(data):
    result = np.array([])
    
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
    
    return result

      
    
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    
    print("Entered here")
    file = request.files['files'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER1, filename)
    file.save(file_path)

    duration = 3 
    test_data, _ = librosa.load(file_path, duration=duration, res_type='kaiser_fast')
    test_features = extract_features(test_data)
    test_features = scaler.transform(test_features.reshape(1, -1))  # Scale the features
    test_features = np.expand_dims(test_features, axis=2)  # Add a dimension for CNN input

    # Make predictions using the trained model
    predictions = model.predict(test_features)
    predicted_class = np.argmax(predictions)

    classes = {0:'Angry', 1:'Calm', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
      
    pred = classes[predicted_class]
    print(pred)
    
    return render_template('after.html', pred_output=pred)  

@app.route('/notebook')
def notebook():
	return render_template('NOtebook.html')


if __name__ == '__main__':
    app.run(debug=False)