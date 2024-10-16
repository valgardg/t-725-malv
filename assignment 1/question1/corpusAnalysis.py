import sys
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('gutenberg', quiet=True)
nltk.download('stopwords', quiet=True)

def check_args():
    if len(sys.argv) < 2:
        print('missing argument: gutenberg corpus file name')
        return True
    return False

def display_analysis(filename, token_no, token_types, types_no_stopwords, ten_most_common, long_types, nouns_with_ation):
    analysis_string = f"""
Text: {filename}
Tokens: {token_no}
Types: {token_types}
Types excluding stop words: {types_no_stopwords}
10 most common tokens: {ten_most_common}
Long types: {long_types}
Nouns ending in 'ation': {nouns_with_ation}
"""

    print(analysis_string)

def main():
    if check_args():
        return
    filename = sys.argv[1]
    print(f'Conducting analysis on file {filename}')
    tokens = gutenberg.words(filename)
    # number of tokens
    token_no = len(tokens)
    # token types
    token_types = len(set(tokens))
    # types with no stopwords
    types_no_stopwords = len(set([t for t in tokens if t not in stopwords.words("english")]))
    # 10 most common
    file_fdist = FreqDist(token for token in tokens)
    # long types
    long_types = list(set([token for token in tokens if len(token) > 13 and '-' not in token]))
    long_types_sorted = sorted(long_types, key=len, reverse=True)
    # nouns ending in 'ation'
    tagged_tokens = nltk.pos_tag(tokens)
    nouns_with_ation = [word for word, pos in tagged_tokens if pos.startswith('NN') and word.endswith('ation')]

    # display analysis
    display_analysis(filename, token_no, token_types, types_no_stopwords, file_fdist.most_common(10), long_types_sorted[:4], nouns_with_ation[:10])

if __name__ == "__main__":
    main()