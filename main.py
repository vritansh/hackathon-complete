#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, request
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import auc, balanced_accuracy_score, make_scorer, roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import joblib


# In[5]:


#Imports for sentiment analysis 
import nltk
nltk.download('vader_lexicon')
import tweepy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[6]:

app = Flask(__name__)

loaded_model = joblib.load('customer_behavior_model.pkl')

# API keyws that yous saved earlier

# api_key = "insert yours"
# api_secrets = "insert yours"
# access_token = "insert yours"
# access_secret = "insert yours"
 
# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key,api_secrets)
auth.set_access_token(access_token,access_secret)
api = tweepy.API(auth)
try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')


# In[8]:


def percentage(part,whole):
     return 100 * float(part)/float(whole)

def fetch_top_user_tweets():
            keyword = "tesla"
            noOfTweet = 10
            # keyword = input('Please enter keyword or hashtag to search: ')
            # noOfTweet = int(input ('Please enter how many tweets to analyze: '))
            tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)
            positive = 0
            negative = 0
            neutral = 0
            polarity = 0
            tweet_list = []
            neutral_list = []
            negative_list = []
            positive_list = []
            positive_user = []
            for tweet in tweets: 
              #print(tweet.text)
              tweet_list.append(tweet.text)
              analysis = TextBlob(tweet.text)
              score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
              neg = score['neg']
              neu = score['neu']
              pos = score['pos']
              comp = score['compound']
              polarity += analysis.sentiment.polarity

              if neg > pos:
                negative_list.append(tweet.text)
                negative += 1
              elif pos > neg:
                positive_list.append(tweet.text)
                positive_user.append(tweet.user.id)
                positive += 1

              elif pos == neg:
                neutral_list.append(tweet.text)
                neutral += 1
            positive = percentage(positive, noOfTweet)
            negative = percentage(negative, noOfTweet)
            neutral = percentage(neutral, noOfTweet)
            polarity = percentage(polarity, noOfTweet)
            positive = format(positive, '.1f')
            negative = format(negative, '.1f')
            neutral = format(neutral, '.1f')
            
            count=0
            dicts = {}
            for user in positive_user:
              count=count+1
              user_tweet = []
              timeline=tweepy.Cursor(api.user_timeline, id=user).items(5)
              for r in timeline:
                user_tweet.append(r.text)
                #print(r.text)
              dicts[user] = user_tweet

#               print("user {}".format(count))


            return dicts


# In[9]:


def make_prediction(new_data):
    # All the columns that are expected by the model
    cols = ['Age', 'EstimatedSalary', 'Gender_Female', 'Profession_Artist',
            'Profession_Doctor', 'Profession_Engineer', 'Profession_Entertainment',
            'Profession_Executive', 'Profession_Healthcare', 'Profession_Homemaker',
            'Profession_Lawyer', 'Profession_Marketing', 'Ever_Married_Yes',
            'Spending_Score_Average', 'Spending_Score_High', 'Spending_Score_Low']

    # Encoding the categorical variables and adding the missing columns
    new_data_enc = pd.get_dummies(new_data, columns=['Gender', 'Profession', 'Ever_Married', 'Spending_Score'])
    new_data_enc = new_data_enc.reindex(columns=cols, fill_value=0)


    # Making predictions
    predictions = loaded_model.predict(new_data_enc)

    return bool(predictions[0])  # Convert 0/1 to True/False


# Sample data
# {
#     "Age": 30,
#     "Gender": "Male",
#     "Purchased": true,
#     "EstimatedSalary": 200000,
#     "Profession": "Entertainment",
#     "Ever_Married": false,
#     "Spending_Score": "High"
# }

@app.route('/generate', methods=['POST'])
def predict():
    data = request.get_json()
  
    age = data.get('Age')
    gender = data.get('Gender')
    purchased = data.get('Purchased')
    estimated_salary = data.get('EstimatedSalary')
    profession = data.get('Profession')
    Ever_Married = data.get('Ever_Married')
    Spending_Score = data.get('Spending_Score')
      
    #twitter_handle = data.get('twitterHandle') 
    
    twitterhandle = data.get('twitter_handle')
    
    new_data = pd.DataFrame({
    'Age': [age],
    'EstimatedSalary': [estimated_salary],
    'Purchased': [purchased],
    'Gender': [gender],
    'Profession': [profession],
    'Ever_Married': [Ever_Married],
    'Spending_Score': [Spending_Score]
    })

    
    prediction =  make_prediction(new_data)
    
    s = fetch_top_user_tweets()
    if(len(list(s.keys())) >0):
        first_user = list(s.keys())[0]
        tweet_1 = s[first_user][0]
    print(tweet_1)
    
    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)