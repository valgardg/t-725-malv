from nltk.corpus import gutenberg
import nltk
nltk.download('gutenberg')

# Required for nltk.pos_tag_sents()
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

from nltk import FreqDist
brown_sents = gutenberg.sents('chesterton-brown.txt')
print(brown_sents[0])
brown_tagged = nltk.pos_tag_sents(brown_sents)
print(brown_tagged[0])
brown_tags = [tags for sents in brown_tagged for _, tags in sents]
print(brown_tags[:10])
brown_fd = FreqDist(brown_tags)
print(brown_fd.most_common())