import nltk
from evaluateParser import evaluate_parser

q2_grammar = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> NP PP | DT ADJ NN | CD ADJ NNS | PRP NN | DT NN | NN
  VP -> V NP | V PP
  PP -> P NP | ADV P NP
  DT -> 'the'
  N -> NN | NNS
  NN -> 'venus' | 'planet' | 'earth' | 'sun' | 'life' | 'saturn'
  NNS -> 'moons'
  CD -> '82'
  ADJ -> 'closest' | 'known'
  ADV -> 'halfway'
  PRP -> "its"
  V -> 'is' | 'has'
  P -> 'to' | 'through'
  """)

q2_parser = nltk.ChartParser(q2_grammar)

# Facts
q2_facts = [
    "venus is the closest planet to earth",
    "the sun is halfway through its life",
    "saturn has 82 known moons"
]

# Evaluate the parser
q2_correct = [
    "(S (NP (NN venus)) (VP (V is) (NP (NP (DT the) (ADJ closest) (NN planet)) (PP (P to) (NP (NN earth))))))",
    "(S (NP (DT the) (NN sun)) (VP (V is) (PP (ADV halfway) (P through) (NP (PRP its) (NN life)))))",
    "(S (NP (NN saturn)) (VP (V has) (NP (CD 82) (ADJ known) (NNS moons))))"
]

evaluate_parser(q2_parser, q2_facts, q2_correct)