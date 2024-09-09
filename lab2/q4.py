import nltk
from nltk.corpus import subjectivity
nltk.download('subjectivity')

objFileid = subjectivity.fileids('obj')
subjFileid = subjectivity.fileids('subj')

sentTokensObj = subjectivity.sents(objFileid)
sentTokensSubj = subjectivity.sents(subjFileid)

# question 2 - generating set with 2000 mcw and feature fucntion

subj_fd = nltk.FreqDist(subjectivity.words())
subj_words = {word for word, count in subj_fd.most_common(2000)}

def get_toksent_features(tokenized_sent):
    sent_words = set(tokenized_sent)
    return {word: word in sent_words for word in subj_words}

# question 3 - generating training and test sets

obj_examples = [(get_toksent_features(obj), 'obj') for obj in sentTokensObj]
subj_examples = [(get_toksent_features(subj), 'subj') for subj in sentTokensSubj]

subjectivity_training = obj_examples[:4500] + subj_examples[:4500]
subjectivity_test = obj_examples[4500:] + subj_examples[4500:]

print('training length:', len(subjectivity_training))
print('test length:', len(subjectivity_test))

# question 4 - naive bayes classifier

subjectivity_classifier = nltk.NaiveBayesClassifier.train(subjectivity_training)
print("Accuracy:", nltk.classify.accuracy(subjectivity_classifier, subjectivity_test))

subjectivity_classifier.show_most_informative_features(20)