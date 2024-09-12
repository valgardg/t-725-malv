import nltk
from nltk.corpus import udhr
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('udhr')

print(udhr.categories())

# vectorizer = CountVectorizer(ngram_range=(1,3), analyzer='char')

# # count all unigrams, bigrams. and trigrams in the sentences and create a feature vector
# sentences = ["It was the best of times, it was the worst of times,",
#              "It was the age of wisdom, it was the age of foolishness,"]

# vector = vectorizer.fit_transform(sentences)

# print("Unigrams, bigrams, and trigrams:", vectorizer.get_feature_names_out())
# print("\nFeatures:")
# print(vector.toarray())

