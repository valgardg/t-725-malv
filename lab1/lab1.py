import nltk
from nltk.corpus import gutenberg

# download gutenberg corpus
nltk.download('gutenberg')

# get plain-text version of moby dick
moby_raw = gutenberg.raw('melville-moby_dick.txt')

# get all tokens in moby dick
moby_tokens = gutenberg.words('melville-moby_dick.txt')

# print first 10 tokens
print("First 10 tockets:", moby_tokens[:10])

# print the first 250 characters of Moby dick
print("Frist 250 characters:", moby_raw[:250])