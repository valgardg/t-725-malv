import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
# download gutenberg corpus
nltk.download('gutenberg')
# download punk_tab
nltk.download('punkt')

# get plain-text version of caroll-alice.txt
alice_raw = gutenberg.raw('carroll-alice.txt')

# get all tokens in alice in wonderland
alice_tokens = word_tokenize(alice_raw)
print("Total number of tokens:", len(alice_tokens))

# get unique tokens
print("Number of unique tokens:", len(set(alice_tokens)))

