from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer #
# used to generate feature vectors containing character or word n-gram counts for any n within a given range ->
# (e.g ngram_range=(2,2) for only bigrams, or ngram_range(1,3) for unigrams, bigrams and trigrams)
# the CountVectorizer has an attribute called 'analyzer' that can be set to 'char' for character n-grams

# initialize a vectorizer that counts word bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))

# count all bigrams in the sentences and create a feature vector
sentences = ["It was the best of times, it was the worst of times,",
             "It was the age of wisdom, it was the age of foolishness,"]

vector = vectorizer.fit_transform(sentences)

print("Bigrams:", vectorizer.get_feature_names_out())
print("\nFeatures:")
print(vector.toarray())