import nltk
from nltk.corpus import subjectivity
nltk.download('subjectivity')

objFileid = subjectivity.fileids('obj')
subjFileid = subjectivity.fileids('subj')

sentTokensObj = subjectivity.sents(objFileid)
setnTokensSubj = subjectivity.sents(subjFileid)

# question 2 - generating set with 2000 mcw and feature fucntion

subj_fd = nltk.FreqDist(subjectivity.words())
subj_words = {word for word, count in subj_fd.most_common(2000)}

def get_toksent_features(tokenized_sent):
    sent_words = set(tokenized_sent)
    return {word: word in sent_words for word in subj_words}

example_features = get_toksent_features(sentTokensObj[0])
# print(example_features)