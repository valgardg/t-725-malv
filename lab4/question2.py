import nltk
from evaluateParser import evaluate_parser

q1_grammar = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> N | NP PP
  PP -> P NP
  VP -> V ADJ
  ADJ -> "blue"
  V -> "are"
  N -> "sunsets" | "mars" 
  P -> "on"
  """)

q1_parser = nltk.ChartParser(q1_grammar)

# Evaluate the parser
q1_sents = ["sunsets on mars are blue"]
q1_correct = ["(S (NP (NP (N sunsets)) (PP (P on) (NP (N mars)))) (VP (V are) (ADJ blue)))"]

evaluate_parser(q1_parser, q1_sents, q1_correct)