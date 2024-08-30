import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
# download gutenberg corpus
nltk.download('gutenberg')
# download punk_tab
nltk.download('punkt')

# get plain-text version of caroll-alice.txt
alice_raw = gutenberg.raw('carroll-alice.txt')

# get all tokens in alice in wonderland
alice_tokens = word_tokenize(alice_raw)

alice_fdist = FreqDist()

for token in alice_tokens:
    alice_fdist[token] += 1

alice_fdist.pprint(20)