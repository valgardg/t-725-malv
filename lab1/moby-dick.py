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
print("First 250 characters:", moby_raw[:250])

# analyze contents of text
moby_text = nltk.Text(moby_tokens)
moby_text.concordance('Iceland')

# download punkt tokenizer model
nltk.download('punkt')

moby_sentences = nltk.sent_tokenize(moby_raw) # split raw text into sentences
tokens = nltk.word_tokenize(moby_sentences[3]) # split string into tokens

print("First 5 sentences:")
for sentence in moby_sentences[:5]:
    print(">>>", sentence)

print(f"\nTotal number of sentences: {len(moby_sentences):,}")

print("\nTokens:", tokens)