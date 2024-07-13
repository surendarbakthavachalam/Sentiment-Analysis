import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = stopwords.words('english')
intel_specific_stopwords = ['core', 'processor', 'intel']
stop_words.extend(intel_specific_stopwords)


analyzer = SentimentIntensityAnalyzer()

def analyze_review(review):

  review_text = review.lower()
  review_text = ''.join([word for word in review_text if word.isalpha() or word == ' '])
  words = [word for word in review_text.split() if word not in stop_words]
  preprocessed_review = ' '.join(words)

  sentiment = analyzer.polarity_scores(preprocessed_review)
  return sentiment, preprocessed_review

review = "intel i3 12th gen.csv"
sentiment, preprocessed_review = analyze_review(review)

compound = sentiment['compound']
pos = sentiment['pos']
neg = sentiment['neg']
neu = sentiment['neu']

labels = 'Positive', 'Negative', 'Neutral'

total = pos + neg + neu
pos_perc = (pos / total) * 100
neg_perc = (neg / total) * 100
neu_perc = (neu / total) * 100

pie_data = [pos_perc, neg_perc, neu_perc]

colors = ['green', 'gold', 'red']  # Customize colors as desired

plt.pie(pie_data, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
plt.title("Sentiment Analysis of Review")
plt.axis('equal')  # Equal aspect ratio ensures a circular pie chart
plt.show()

print("Original Review:", review)
print("Preprocessed Review:", preprocessed_review)
print("Sentiment Scores:", sentiment)
