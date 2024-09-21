import nltk

q1_grammar = nltk.PCFG.fromstring("""
  S -> NP VP [1.0]
  NP -> N [0.5]
  NP ->  NP PP [0.3] 
  NP -> DT N [0.2]
  VP -> V NP [0.8]
  VP -> V NP PP [0.2]
  PP -> P NP [1.0]
  V -> "saw" [1.0]
  DT -> "a" [1.0]
  N -> "John" [0.6]
  N -> "man" [0.3]
  N -> "binoculars" [0.1]
  P -> "with" [1.0]
  """)

sentence = "John saw a man with binoculars".split()

q1_parser = nltk.ViterbiParser(q1_grammar)
trees = q1_parser.parse(sentence)

for tree in trees:
  print(tree)
  tree.pretty_print()
  print()