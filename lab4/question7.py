import nltk
import svgling
from nltk.corpus import conll2000
from question6 import TrigramChunker
from question6 import test_sents

tagged_sents = conll2000.tagged_sents('test.txt')

triChunker = TrigramChunker(tagged_sents)
sent1 = triChunker.parse(tagged_sents[0])
correctSent1 = test_sents[0]

svgling.draw_tree(correctSent1)

# Your solution here
tagged_sents = conll2000.tagged_sents('test.txt')

# Sentence 1
sent1 = triChunker.parse(tagged_sents[5])
correctSent1 = test_sents[5]
svgling.draw_tree(sent1)

# an example of a mistake for this sentence appears to be every single word... not exactly sure what is going on for it to be seemingly wrong on every word

# Sentence 2
sent2 = triChunker.parse(tagged_sents[10])
correctSent2 = test_sents[10]
svgling.draw_tree(sent2)

# again, it seems like the trees just dont add up at all

# Sentence 3
sent3 = triChunker.parse(tagged_sents[15])
correctSent3 = test_sents[15]
svgling.draw_tree(sent3)

# same as before