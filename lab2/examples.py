import nltk
from nltk.corpus import movie_reviews
nltk.download('punkt')

nltk.download('movie_reviews')
print("Categories:", movie_reviews.categories())

pos_fileids = movie_reviews.fileids('pos')
neg_fileids = movie_reviews.fileids('neg')

print(pos_fileids[:5])  # The first 5 positive reviews
print(neg_fileids[:5])  # The first 5 negative reviews

pos_reviews = [movie_reviews.words(fid) for fid in pos_fileids]
neg_reviews = [movie_reviews.words(fid) for fid in neg_fileids]

print(pos_reviews[0][:10])  # The first 10 tokens of the first positive review
print(neg_reviews[0][:10])  # The first 10 tokens of the first negative review

# Create a set with 2,000 of the most frequent words in the movie review corpus
movie_fd = nltk.FreqDist(movie_reviews.words())
movie_words = {word for word, count in movie_fd.most_common(2000)}

# For a given review (in the form of a list or set of tokens), create a
# dictionary which tells us which words are present and which are not.
def get_review_features(review):
  review_words = set(review)
  return {word: word in review_words for word in movie_words}

# Let's see how this works for the first positive review:
example_features = get_review_features(pos_reviews[0])
print("'funny' is in the review:", example_features['funny'])
print("'boring' is in the review:", example_features['boring'])

pos_examples = [(get_review_features(review), 'pos') for review in pos_reviews]
neg_examples = [(get_review_features(review), 'neg') for review in neg_reviews]

movie_training = pos_examples[:900] + neg_examples[:900]  # 1800 examples total
movie_test = pos_examples[900:] + neg_examples[900:]  # 200 examples total

movie_classifier = nltk.NaiveBayesClassifier.train(movie_training)

print("Accuracy:", nltk.classify.accuracy(movie_classifier, movie_test))

movie_classifier.show_most_informative_features(20)

