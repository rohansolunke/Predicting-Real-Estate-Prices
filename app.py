import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

data = pd.read_csv("cleaned_data.csv")
model2 = pickle.load(open('model2.pkl', 'rb'))

X = data.drop('price', axis='columns')
y = data['price']

@app.route('/') #index or landing page of website
def home():
    return render_template('new_house.html')
# 127.0.0.1:8080/predict
@app.route('/predict', methods=['GET','POST']) #post method is used to send parameters in http request
def predict():
    '''
    For rendering results on HTML GUI'''
    location = request.form.get("location")
    sqft = int(request.form.get("sq_ft"))
    b = int(request.form.get("bhk"))
    
    

    new = X.iloc[:,4:100]
    def prediction(location, bhk, bath, balcony, sqft):
    
        loc_index= -1
        
        if location in (new):
            loc_index = int(np.where(X.columns==location)[0][0])
            
        x = np.zeros(len(X.columns))
        x[0] = bath
        x[1] = balcony
        x[2] = bhk
        x[3] = sqft
    
        if loc_index >= 0:
            x[loc_index] = 1 
        if x[3] in range(300,10000):
            x[3] = sqft
        else:
            return False
        return model2.predict([x])[0]
    
    # l,s,bh = location,sqft,b
    # Input in the form : Location, BHK, Bath, Balcony, Sqft.
    
    pred = prediction(location,b, 2, 2, sqft)

    # output = round(pred, 2)
    if pred == False:
        return render_template('new_house.html',
    	prediction_text='Enter sqft in range (300-5000)')
    else:
        return render_template('new_house.html',
    	    prediction_text='Estimated Price is {} lakhs'
            .format(round(pred,2)))


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080) # EC2 on AWS
