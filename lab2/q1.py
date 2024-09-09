import nltk
from nltk.corpus import subjectivity
nltk.download('subjectivity')
print('Categories:', subjectivity.categories())

objFileid = subjectivity.fileids('obj')
subjFileid = subjectivity.fileids('subj')

print('Relative path of obj:', objFileid)
print('Relative path of subj:', subjFileid)

sentTokensObj = subjectivity.sents(objFileid)
setnTokensSubj = subjectivity.sents(subjFileid)

print('number of tokenized sentences for objective category:', len(sentTokensObj))
print('number of tokenized sentences for subjective category:', len(setnTokensSubj))

print('first 5 objective tokenized sentences:', sentTokensObj[:5])
print('first 5 subjective tokenized sentences:', setnTokensSubj[:5])