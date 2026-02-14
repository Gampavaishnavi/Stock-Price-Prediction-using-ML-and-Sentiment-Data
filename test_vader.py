from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
print("good:", analyzer.polarity_scores("good"))
print("bad:", analyzer.polarity_scores("bad"))
print("crashing:", analyzer.polarity_scores("crashing"))
