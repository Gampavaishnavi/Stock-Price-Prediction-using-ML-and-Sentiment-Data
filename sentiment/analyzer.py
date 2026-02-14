from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            # Test VADER
            if self.analyzer.polarity_scores("good")['compound'] == 0:
                print("VADER returned 0 for 'good'. Fallback to keyword mode.")
                self.use_vader = False
            else:
                self.use_vader = True
        except Exception as e:
            print(f"VADER init failed: {e}. Fallback to keyword mode.")
            self.use_vader = False

        self.positive_words = {'good', 'great', 'bull', 'bullish', 'rise', 'gain', 'profit', 'up', 'buy', 'growth'}
        self.negative_words = {'bad', 'terrible', 'bear', 'bearish', 'fall', 'loss', 'down', 'sell', 'crash', 'risk'}

        # NewsAPI Client
        self.news_api_key = os.getenv("NEWS_API_KEY") 
        self.newsapi = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None

    def get_sentiment(self, text):
        """Returns the compound sentiment score for a given text."""
        if not text:
            return 0.0
            
        if self.use_vader:
            try:
                score = self.analyzer.polarity_scores(str(text))['compound']
                return score
            except:
                pass
        
        # Keyword Fallback
        words = str(text).lower().split()
        score = 0
        for word in words:
            if word in self.positive_words:
                score += 1
            elif word in self.negative_words:
                score -= 1
        
        if score > 0: return min(score / 3.0, 1.0)
        elif score < 0: return max(score / 3.0, -1.0)
        return 0.0

    def fetch_news_sentiment(self, ticker):
        """
        Fetches real-time news sentiment.
        Priority 1: NewsAPI (if API Key exists)
        Priority 2: FinViz Scraping (Free fallback)
        Returns: (average_score, list_of_headlines)
        """
        headlines = []
        
        # 1. Try NewsAPI
        if self.newsapi:
            try:
                # Get news about the ticker
                articles = self.newsapi.get_everything(q=ticker,
                                                     language='en',
                                                     sort_by='publishedAt',
                                                     page_size=10)
                if articles['status'] == 'ok':
                    for article in articles['articles']:
                        headlines.append(article['title'])
                    print(f"Fetched {len(headlines)} articles from NewsAPI for {ticker}")
            except Exception as e:
                print(f"NewsAPI error: {e}")

        # 2. Keyless Fallback: FinViz Scraping
        if not headlines:
            print(f"Attempting FinViz scraping for {ticker}...")
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                req = requests.get(url, headers=headers)
                soup = BeautifulSoup(req.content, 'html.parser')
                news_table = soup.find(id='news-table')
                
                if news_table:
                    # Get most recent 10 headlines
                    rows = news_table.findAll('tr')
                    for row in rows[:10]:
                        headline = row.a.text
                        headlines.append(headline)
            except Exception as e:
                print(f"FinViz scraping error: {e}")

        # 3. Analyze Headlines
        if headlines:
            scores = [self.get_sentiment(h) for h in headlines]
            avg_score = np.mean(scores)
            return avg_score, headlines
        else:
            return 0.0, ["No news found."]

    def simulate_daily_sentiment(self, start_date, end_date):
        """Generates simulated daily sentiment scores for a date range (for backtesting)."""
        dates = pd.date_range(start=start_date, end=end_date)
        n = len(dates)
        np.random.seed(42)
        sentiment_changes = np.random.normal(0, 0.1, n)
        sentiment_trend = np.cumsum(sentiment_changes)
        if np.max(np.abs(sentiment_trend)) > 0:
             sentiment_trend = 0.8 * sentiment_trend / np.max(np.abs(sentiment_trend))
        noise = np.random.normal(0, 0.2, n)
        final_scores = np.clip(sentiment_trend + noise, -1, 1)
        return pd.DataFrame({'Date': dates, 'Sentiment': final_scores})

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("Test Sentiment:", analyzer.get_sentiment("Market is bullish on tech"))
    print("\n--- Testing Live Fetch (AAPL) ---")
    score, headlines = analyzer.fetch_news_sentiment("AAPL")
    print(f"Score: {score:.2f}")
    for h in headlines[:3]:
        print(f"- {h}")
