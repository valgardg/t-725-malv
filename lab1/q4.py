import re
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize

# get plain-text version of caroll-alice.txt
alice_raw = gutenberg.raw('carroll-alice.txt')

# get all tokens in alice in wonderland
alice_tokens = word_tokenize(alice_raw)

alice_text = nltk.Text(alice_tokens)

alice_text.findall(r'<.*x.*ed>')