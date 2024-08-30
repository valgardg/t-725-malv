import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Ensure that the necessary resources are downloaded
nltk.download('gutenberg')
nltk.download('punkt')

# Load the raw text of Alice in Wonderland
alice_raw = gutenberg.raw('carroll-alice.txt')

# Tokenize the text
alice_tokens = word_tokenize(alice_raw)

# Create a frequency distribution
fdist = FreqDist(alice_tokens)

# Get the 20 most common tokens
most_common_tokens = fdist.most_common(20)

# Print the results
print("20 Most Common Tokens in 'Alice in Wonderland':")
for token, frequency in most_common_tokens:
    print(f"Token: {token}, Frequency: {frequency}")
