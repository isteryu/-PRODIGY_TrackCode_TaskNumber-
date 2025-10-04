import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re 

import kagglehub
from kagglehub import KaggleDatasetAdapter


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError: 
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

sns.set_style('whitegrid')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
try:
    file_path = "twitter_sentiment_entity.csv" 
    
    
    print("Attempting to load dataset from Kaggle Hub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jp797498e/twitter-entity-sentiment-analysis",
        file_path,
    )
    
    
    df.columns = ['Entity', 'True_Sentiment', 'Tweet']
    df = df[['Tweet', 'True_Sentiment', 'Entity']] 

    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset from Kaggle Hub: {e}")
    print("Please ensure you have installed 'kagglehub' and are authenticated/connected.")
    exit()

print("\n--- Initial Data Head ---")
print(df[['Tweet', 'True_Sentiment', 'Entity']].head())
print(f"Dataset shape: {df.shape}")

def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www.\S+', '', text)

    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text) 
    return text.strip().lower()

df['Cleaned_Tweet'] = df['Tweet'].apply(clean_tweet)

def get_vader_scores(text):
    if pd.isna(text):
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    return sia.polarity_scores(text)


df['VADER_Scores'] = df['Cleaned_Tweet'].apply(get_vader_scores)

df['Compound_Score'] = df['VADER_Scores'].apply(lambda x: x['compound'])


def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['VADER_Sentiment'] = df['Compound_Score'].apply(classify_sentiment)

print("\n--- Sentiment Analysis Results Snapshot ---")
print(df[['Cleaned_Tweet', 'True_Sentiment', 'VADER_Sentiment', 'Entity']].head())




plt.figure(figsize=(18, 7))
plt.suptitle('Sentiment Analysis of Twitter Data', fontsize=18, weight='bold')

plt.subplot(1, 3, 1)
sentiment_counts = df['VADER_Sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['salmon', 'skyblue', 'lightgreen'])
plt.title('1. Distribution of Calculated Sentiments')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.xticks(rotation=0)


plt.subplot(1, 3, 2)

df['True_Sentiment'] = df['True_Sentiment'].str.strip()
sentiment_comparison = df.groupby('True_Sentiment')['Compound_Score'].mean().reset_index()
sns.barplot(x='True_Sentiment', y='Compound_Score', data=sentiment_comparison, palette='viridis')
plt.title('2. Mean VADER Score by Original Label')
plt.xlabel('Original Sentiment Label')
plt.ylabel('Mean VADER Compound Score')
plt.xticks(rotation=45)


plt.subplot(1, 3, 3)
entity_sentiment = df.groupby(['Entity', 'VADER_Sentiment']).size().unstack(fill_value=0)
entity_sentiment_norm = entity_sentiment.div(entity_sentiment.sum(axis=1), axis=0)
entity_sentiment_norm.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='coolwarm')
plt.title('3. Sentiment Distribution by Entity/Brand')
plt.xlabel('Entity/Brand')
plt.ylabel('Proportion of Tweets')
plt.legend(title='VADER Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n--- Key Sentiment Insights ---")
print("1. **Overall Sentiment:** The distribution of Positive, Negative, and Neutral tweets is shown in the first bar chart.")
print("2. **VADER vs. True:** The second chart validates that the positive true labels generally have higher VADER scores, and negative labels have lower scores.")
print("3. **Brand/Entity Analysis:** The stacked bar chart visualizes the public's attitude (proportion of positive/negative/neutral) towards specific entities present in the dataset, such as specific companies or products.")
