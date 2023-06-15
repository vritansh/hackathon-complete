# importing libraries
import nltk
nltk.download('vader_lexicon')
import tweepy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# API keyws that yous saved earlier
api_key = "TxzOEBKMfA7HY2Vf37hrWLo3V"
api_secrets = "XNXJmrARDgAoUxPZi3IqlSQbDVjZjTKRd6Qiw7y0LJuyJlVd3C"
access_token = "241533436-nN0A7PdYXlv41FVYb40NKCi0N0Yi9JO3Yw0AdGFG"
access_secret = "EvYBNpXMxy35v1DkmICWiJDgMiwL3RWhhBaSuHRxPcmT2"

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


# In[12]:


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
            




