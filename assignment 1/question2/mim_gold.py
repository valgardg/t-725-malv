import nltk
from nltk.corpus.reader import TaggedCorpusReader
from nltk.probability import FreqDist
from nltk import bigrams
from nltk.probability import ConditionalFreqDist

MIM_DIRECTORY_NAME = 'MIM'
CORPUS_PATTERN = '.*.sent'
punctuation = list('.,')

def read_mim():
    mim_gold_corpus = TaggedCorpusReader(MIM_DIRECTORY_NAME, CORPUS_PATTERN)
    return mim_gold_corpus

def t1(mim_gold_corpus):
    sentences = mim_gold_corpus.sents()

    sentence_no = len(sentences)
    sentence_100 = sentences[99]

    # print(f'Task 1:')
    print(f'Number of sentences: {sentence_no}')
    print(f'Sentence no. 100:\n{" ".join(sentence_100)}')
    print('\n')

def t2(mim_gold_corpus):
    tokens = mim_gold_corpus.words()
    token_no = len(tokens)
    types_no = len(set(tokens))

    # print(f'Task 2:')
    print(f'Number of tokens: {token_no}')
    print(f'Number of types: {types_no}')
    print('\n')

def t3(mim_gold_corpus):
    tokens = mim_gold_corpus.words()
    mim_freq_dist = FreqDist(token for token in tokens)
    top_10 = "\n".join([f"{token[0]} => {token[1]}" for token in mim_freq_dist.most_common(10)])
    # print(f'Task 3:')
    print(f'The 10 most frequent tokens: \n{top_10}')
    print('\n')

def t4(mim_gold_corpus):
    tagged_words = mim_gold_corpus.tagged_words()
    tags = [tag for (word, tag) in tagged_words]
    tag_freq_dist = FreqDist(tag for tag in tags)
    top_20 = "\n".join([f"{tag[0]} => {tag[1]}" for tag in tag_freq_dist.most_common(20)])
    # print(f'Task 4:')
    print(f'The 20 most frequent PoS tags: \n{top_20}')
    print('\n')

def t5(mim_gold_corpus):
    tagged_words = mim_gold_corpus.tagged_words()
    tag_bigrams = bigrams(tagged_words)

    cfd = ConditionalFreqDist((first_tag, second_tag) for ((_, first_tag), (_, second_tag)) in tag_bigrams if first_tag == 'AF')
    most_common_10 = "\n".join(f"{tag[0]} => {tag[1]}" for tag in cfd['AF'].most_common(10))

    # print(f'Task 5:')
    print(f"The 10 most frequent PoS tags following the tag 'af': \n{most_common_10}\n")

def main():
    mim_gold_corpus = read_mim()
    t1(mim_gold_corpus)
    t2(mim_gold_corpus)
    t3(mim_gold_corpus)
    t4(mim_gold_corpus)
    t5(mim_gold_corpus)

if __name__ == "__main__":
    main()