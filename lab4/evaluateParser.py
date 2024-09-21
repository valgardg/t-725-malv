# Compare trees generated by a parser to correct trees
import nltk
def evaluate_parser(parser, sentences, correct):
  correct_trees = [nltk.Tree.fromstring(t) for t in correct]
  num_sentences = len(sentences)

  for num, (sent, correct_tree) in enumerate(zip(sentences, correct_trees)):
    print(f">{sent}\n")

    error = None
    print_tree = False

    try:
      my_trees = list(parser.parse(sent.split()))
    except ValueError as e:
      print(e)
      my_trees = []

    if len(my_trees) > 1:
      error = f"Generated {len(my_trees)} trees (should only generate one)!"
    elif len(my_trees) == 0:
      error = "Couldn't parse sentence."
    else:
      my_tree = my_trees[0]
      if my_tree != correct_tree:
        error = "Generated an incorrect tree"
        print_tree = True

    if error:
      print(error)

      if print_tree:
        print("\nYour tree:")
        my_tree.pretty_print()

      print("\nCorrect tree:")
      correct_tree.pretty_print()
    else:
      print("Correct!\n")

    if num != num_sentences - 1:
      print("============================")