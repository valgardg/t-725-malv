import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")

def find_similar(a):
  # 5 most similar to a
  a = a.lower()
  print(f"> 5 most similar to {a}:?")
  top_words = glove.most_similar(a, topn=5)
  for num, (word, score) in enumerate(top_words):
    print(f"{num + 1}: ({score:.3f}) {word}")
  print()

# find 5 most similar to cat
find_similar('cat')
# find 5 most similar to samsung
find_similar('samsung')
# find 5 most similar to batman
find_similar('batman')

def doesnt_match(words):
  lowered_words = [word.lower() for word in words]
  print(f"> What word doesnt fit from {lowered_words}")
  doesnt_fit = glove.doesnt_match(lowered_words)
  print(f"the word '{doesnt_fit}' does not fit in the list")
  print()

doesnt_match("cat hamster gremlin rabbit goldfish dog".split())
doesnt_match("samsung microsoft dell panasonic mcdonalds facebook".split())
doesnt_match("batman spiderman daredevil shrek hulk deadpool".split())