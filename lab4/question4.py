import nltk
from evaluateParser import evaluate_parser

q3_grammar = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> DT N N | N | CD
  VP -> IV PP PP | TV PP PP | TV NP PP | TV NP PP
  PP -> P | P NP
  DT -> 'the'
  N -> 'curiosity' | 'rover' | 'mars' | 'nasa' | 'perseverence'
  TV -> 'launched'
  IV -> 'arrived'
  P -> 'on' | 'in'
  CD -> '2012' | '2020'
  """)

q3_parser = nltk.ChartParser(q3_grammar)

# Facts
q3_facts = [
    # "the curiosity rover arrived on mars in 2012",
    "nasa launched the perseverence rover in 2020"
]

# Evaluate the parser
q3_correct = [
    # "(S (NP (DT the) (N curiosity) (N rover)) (VP (IV arrived) (PP (P on) (NP (N mars))) (PP (P in) (NP (CD 2012)))))",
    "(S (NP (N nasa)) (VP (TV launched) (NP (DT the) (N perseverence) (N rover)) (PP (P in) (NP (CD 2020)))))"
]

evaluate_parser(q3_parser, q3_facts, q3_correct)