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
from dotenv import load_dotenv
import nltk
nltk.download('vader_lexicon')
import tweepy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv, find_dotenv
import os

# Load the .env file
load_dotenv(find_dotenv())

# Get the API key and secret
api_key = os.environ.get("TWITTER_API_KEY")
api_secrets = os.environ.get("TWITTER_API_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_secret = os.environ.get("TWITTER_ACCESS_SECRET")



# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key,api_secrets)
auth.set_access_token(access_token,access_secret)
# Set api as a global variable
global api
api = tweepy.API(auth)


try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')

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

    return dicts


import joblib
import pandas as pd
loaded_model = joblib.load('customer_behavior_model.pkl')
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


import openai
def generate_customized_email(data, Stock_name = 'Tesla'):
    '''
    Using the variables in the data pandas dataframe, this function will return a customized email
    The variables are used as a prompt in openAI's GPT-4 to generate the email
    Input: data - pandas dataframe
    '''

    # There is only one row in the dataframe
    row = data.iloc[0]

    if row['Prediction'] == False:
      return

    prompt = """
        Act like a professional sales email writer. The purpose is to sell a particular stock to customer.
        The customer is a stock investor and you are trying to sell a particular stock to him/her.
        Understand the customer behavior from the details and tweets provided.
        The email should be personalized and should be written in a professional manner.
        There should be some content about the stock and the company.
        The email should be written in a way that the customer is convinced to buy the stock.
        The email should contain the following details:\n
    """
    prompt += "Stock Name: " + Stock_name + "\n"
    prompt += "Customer Age: " + str(row['Age']) + "\n"
    prompt += "Customer Estimated Salary: " + str(row['EstimatedSalary']) + "\n"
    prompt += "Customer Gender: " + str(row['Gender']) + "\n"
    prompt += "Customer Profession: " + str(row['Profession']) + "\n"
    prompt += "Customer Ever Married: " + str(row['Ever_Married']) + "\n"
    prompt += "Customer Spending Score: " + str(row['Spending_Score']) + "\n"
    prompt += "Customer Tweets: " + str(row['Tweets']) + "\n"

    messages= [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model = os.environ.get("OPENAI_MODEL"),
        messages = messages,
        temperature = 0.3
    )
    return response.choices[0].message["content"]