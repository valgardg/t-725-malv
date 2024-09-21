from nltk import Tree
t = Tree.fromstring('(S (NP this tree) (VP (V is) (ADJ pretty)))')

# NLTK won't draw the tree if we're using Google Colab
# t.draw()

# But we can still print a text diagram
t.pretty_print()