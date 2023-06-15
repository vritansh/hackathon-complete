#!/usr/bin/env python
# coding: utf-8


from flask import Flask, request, render_template, jsonify
import pandas as pd
from utils import *


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
  
    age = int(data.get('Age'))
    gender = data.get('Gender')
    purchased = bool(data.get('Purchased'))
    estimated_salary = float(data.get('EstimatedSalary'))
    profession = data.get('Profession')
    Ever_Married = bool(data.get('Ever_Married'))
    Spending_Score = data.get('Spending_Score')
      
    twitterhandle = data.get('twitter_handle')
    
    final_data = pd.DataFrame({
        'Age': [age],
        'EstimatedSalary': [estimated_salary],
        'Purchased': [purchased],
        'Gender': [gender],
        'Profession': [profession],
        'Ever_Married': [Ever_Married],
        'Spending_Score': [Spending_Score]
    })

    
    prediction =  make_prediction(final_data)
    
    s = fetch_top_user_tweets()
    if(len(list(s.keys())) >0):
        first_user = list(s.keys())[0]
        tweet_1 = s[first_user][0]
        final_data['Tweets'] = tweet_1

    # Add prediction and tweet to the data
    final_data['Prediction'] = prediction
    if prediction == False:
        return "This customer is not likely to buy the product"

    email_gen = generate_customized_email(final_data)
    # print('Final Email:', email_gen)
    return jsonify({'email': email_gen})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)