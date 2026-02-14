from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            # Test if VADER is working - simple check
            if self.analyzer.polarity_scores("good")['compound'] == 0:
                 # This might happen if lexicon is missing
                print("VADER returned 0 for 'good'. Fallback to keyword mode.")
                self.use_vader = False
            else:
                self.use_vader = True
        except Exception as e:
            print(f"VADER init failed: {e}. Fallback to keyword mode.")
            self.use_vader = False

        self.positive_words = {'good', 'great', 'bull', 'bullish', 'rise', 'gain', 'profit', 'up', 'buy', 'growth'}
        self.negative_words = {'bad', 'terrible', 'bear', 'bearish', 'fall', 'loss', 'down', 'sell', 'crash', 'risk'}

    def get_sentiment(self, text):
        """
        Returns the compound sentiment score for a given text.
        """
        if not text:
            return 0.0
            
        if self.use_vader:
            try:
                score = self.analyzer.polarity_scores(text)['compound']
                # If VADER returns 0 (neutral), we might want to check for specific financial keywords
                # But for now, let's trust VADER or 0.
                return score
            except:
                pass
        
        # Keyword Fallback
        words = text.lower().split()
        score = 0
        for word in words:
            if word in self.positive_words:
                score += 1
            elif word in self.negative_words:
                score -= 1
        
        # Normalize to -1 to 1 (soft sigmoid-ish)
        if score > 0:
            return min(score / 3.0, 1.0)
        elif score < 0:
            return max(score / 3.0, -1.0)
        return 0.0

    def simulate_daily_sentiment(self, start_date, end_date):
        """
        Generates simulated daily sentiment scores for a date range.
        Uses a random walk with momentum to simulate market sentiment trends.
        """
        dates = pd.date_range(start=start_date, end=end_date)
        n = len(dates)
        
        np.random.seed(42)  # For reproducibility
        
        # Create a sentiment trend (random walk)
        sentiment_changes = np.random.normal(0, 0.1, n)
        sentiment_trend = np.cumsum(sentiment_changes)
        
        # Normalize trend to be within -0.8 to 0.8 to allow noise to not clip too much
        if np.max(np.abs(sentiment_trend)) > 0:
             sentiment_trend = 0.8 * sentiment_trend / np.max(np.abs(sentiment_trend))
        
        # Add daily noise
        noise = np.random.normal(0, 0.2, n)
        
        final_scores = sentiment_trend + noise
        
        # Clip to -1 to 1
        final_scores = np.clip(final_scores, -1, 1)
        
        df = pd.DataFrame({'Date': dates, 'Sentiment': final_scores})
        return df

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("Test Sentiment (good):", analyzer.get_sentiment("market is good"))
    print("Test Sentiment (bad):", analyzer.get_sentiment("market crashing"))
    
    # Test simulation
    print("Testing Simulation...")
    sim_df = analyzer.simulate_daily_sentiment('2023-01-01', '2023-01-10')
    print(sim_df.head())
