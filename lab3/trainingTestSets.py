import nltk
from nltk.corpus import subjectivity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Download the subjectivity corpus and get the sentences for each category
nltk.download('subjectivity')

obj_fileids = subjectivity.fileids('obj')
subj_fileids = subjectivity.fileids('subj')

# Let's get the untokenized sentences from each category
obj_sentences = subjectivity.raw(obj_fileids).splitlines()
subj_sentences = subjectivity.raw(subj_fileids).splitlines()

X = obj_sentences + subj_sentences
y = ['obj'] * 5000 + ['subj'] * 5000

# Create a word unigram count vectorizer and generate the feature vectors
vectorizer = CountVectorizer(ngram_range=(1, 1))
X_vectorized = vectorizer.fit_transform(X)

# Create a training and test set (80%/20% split). This function always shuffles
# the examples before making the split, but we can make sure that it always
# shuffles them the same way by specifying a specific random_state value.
X_train, X_test, y_train, y_test = train_test_split(X_vectorized,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)