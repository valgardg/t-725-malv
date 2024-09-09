import nltk
from nltk.corpus import subjectivity
nltk.download('subjectivity')

objFileid = subjectivity.fileids('obj')
subjFileid = subjectivity.fileids('subj')

sentTokensObj = subjectivity.sents(objFileid)
sentTokensSubj = subjectivity.sents(subjFileid)

subj_fd = nltk.FreqDist(subjectivity.words())
subj_words = {word for word, count in subj_fd.most_common(2000)}

def get_toksent_features(tokenized_sent):
    sent_words = set(tokenized_sent)
    return {word: word in sent_words for word in subj_words}

obj_examples = [(get_toksent_features(obj), 'obj') for obj in sentTokensObj]
subj_examples = [(get_toksent_features(subj), 'subj') for subj in sentTokensSubj]

subjectivity_training = obj_examples[:4500] + subj_examples[:4500]
subjectivity_test = obj_examples[4500:] + subj_examples[4500:]

print('training length:', len(subjectivity_training))
print('test length:', len(subjectivity_test))