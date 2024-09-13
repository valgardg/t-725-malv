import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")

def find_word(a, b, x):
  # a is to b as x is to ?
  a = a.lower()
  b = b.lower()
  x = x.lower()
  print(f"> {a}:{b} as {x}:?")
  top_words = glove.most_similar_cosmul(positive=[x, b], negative=[a])
  for num, (word, score) in enumerate(top_words[:5]):
    print(f"{num + 1}: ({score:.3f}) {word}")
  print()

# Example 1: man is to king as woman is to ?
find_word('man', 'king', 'woman')

# Example 2: evening is to dinner as noon is to ?
find_word('evening', 'dinner', 'noon')

# 1) In the UK, people say 'petrol' instead of 'gas'. Find the British English
# equivalent of 'truck'.
find_word('american', 'truck', 'british')

# 2) Find the capital of France. Remember to use only lowercase characters.
find_word('iceland', 'reykjavik', 'france')

# 3) Find the present tense of the verb "flew".
find_word('drive', 'drove', 'fly')