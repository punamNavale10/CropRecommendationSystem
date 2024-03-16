import numpy as np
from flask import Flask,request,render_template
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorus'])
    K = int(request.form['Potassium'])
    temperature = float(request.form['Temprature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    featured_list =[N,P,K,temperature,humidity,ph,rainfall]
    single_pred = np.array(featured_list).reshape(1,-1)

    prediction = model.predict(single_pred)

    crop_dict = {
        1: 'rice',
        2: 'maize',
        3: 'jute',
        4: 'cotton',
        5: 'coconut',
        6: 'papaya',
        7: 'orange',
        8: 'apple',
        9: 'muskmelon',
        10: 'watermelon',
        11: 'grapes',
        12: 'mango',
        13: 'banana',
        14: 'pomegranate',
        15: 'lentil',
        16: 'blackgram',
        17: 'mungbean',
        18: 'mothbeans',
        19: 'pigeonpeas',
        20: 'kidneybeans',
        21: 'chickpea',
        22: 'coffee'
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is a best crop to be cultivated ".format(crop)
    else:
        result = "Sorry ,we could not determine the best crop"

    return render_template('index.html',result=result)
# python main

if __name__=="__main__":
    app.run(debug=True)

