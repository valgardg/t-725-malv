import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

# Ensure that the necessary resources are downloaded
nltk.download('gutenberg')
nltk.download('punkt')

# Load the raw text of Alice in Wonderland
alice_raw = gutenberg.raw('carroll-alice.txt')

# get sentence tokens
alice_sents = sent_tokenize(alice_raw)

# sort the sentences
sorted_alice_sents = sorted(alice_sents, key=len, reverse=True)

print("Longest sentence:", sorted_alice_sents[0])
# print("Shortest sentence:", sorted_alice_sents[-1])