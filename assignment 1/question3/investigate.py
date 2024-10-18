import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

# save embeddings
def save_embeddings():
    wiki_glove = api.load('glove-wiki-gigaword-100')
    wiki_glove.save("wiki_glove.kv")  # Save to a file with .kv extension
    
    twitter_glove = api.load('glove-twitter-100')
    twitter_glove.save("twitter_glove.kv")
    print("Embeddings saved to disk.")

# Load embeddings once
def load_saved_embeddings():
    wiki_glove = KeyedVectors.load("wiki_glove.kv")  # Load the saved .kv file
    twitter_glove = KeyedVectors.load("twitter_glove.kv")
    return {'wiki': wiki_glove, 'twitter': twitter_glove}

# from lab 3
def find_word(glove, a, b, x):
  # a is to b as x is to ?
  a = a.lower()
  b = b.lower()
  x = x.lower()
  print(f"> {a}:{b} as {x}:?")
  top_words = glove.most_similar_cosmul(positive=[x, b], negative=[a])
  for num, (word, score) in enumerate(top_words[:5]):
    print(f"{num + 1}: ({score:.3f}) {word}")
  print()

# from lab 3
def most_similar(glove, a, n=5):
  # n most similar to a
  a = a.lower()
  print(f"> {n} most similar to {a}:?")
  top_words = glove.most_similar(a, topn=n)
  for num, (word, score) in enumerate(top_words):
    print(f"{num + 1}: ({score:.3f}) {word}")

# Check bias by comparing word similarities
def word_similarity(model, word1, word2, targets):
    similarities_word1 = [model.similarity(word1, target) for target in targets]
    similarities_word2 = [model.similarity(word2, target) for target in targets]
    return similarities_word1, similarities_word2

def get_gender_similarities(embeddings):
    profession_words = ["doctor", "nurse", "scientist", "teacher"]
    # Wiki Glove Bias check
    male_sim_wiki, female_sim_wiki = word_similarity(embeddings['wiki'], "man", "woman", profession_words)

    # Twitter Glove Bias check
    male_sim_twitter, female_sim_twitter = word_similarity(embeddings['twitter'], "man", "woman", profession_words)

    print('Wiki bias:')
    for i in range(len(profession_words)):
        print(f'{profession_words[i]}:\n m - {male_sim_wiki[i]:.2f} f - {female_sim_wiki[i]:.2f} - higher: {"male" if male_sim_wiki[i] > female_sim_wiki[i] else "female"}')

    print('Twitter bias:')
    for i in range(len(profession_words)):
        print(f'{profession_words[i]}:\n m - {male_sim_twitter[i]:.2f} f - {female_sim_twitter[i]:.2f} - higher: {"male" if male_sim_twitter[i] > female_sim_twitter[i] else "female"}')


def find_gender_words(embeddings):
    print(f'Wiki gender find words:')
    find_word(embeddings["wiki"], "he", "she", "doctor")
    find_word(embeddings["wiki"], "she", "he", "doctor")
    find_word(embeddings["wiki"], "boy", "girl", "engineer")
    find_word(embeddings["wiki"], "girl", "boy", "engineer")

    print('\n\n')

    print(f'Twitter gender find words:')
    find_word(embeddings["twitter"], "he", "she", "doctor")
    find_word(embeddings["twitter"], "she", "he", "doctor")
    find_word(embeddings["twitter"], "boy", "girl", "engineer")
    find_word(embeddings["twitter"], "girl", "boy", "engineer")

# Investigate bias in relationship between gender and profession
def check_gender_bias(embeddings):
    get_gender_similarities(embeddings)
    # find_gender_words(embeddings)

def get_so_similar_words(embeddings):
    print(f'Wiki embedding:')
    most_similar(embeddings['wiki'], 'gay', 10)
    most_similar(embeddings['wiki'], 'lesbian', 10)
    most_similar(embeddings['wiki'], 'straight', 10)

    print('\n\n')

    print(f'Twitter embedding:')
    most_similar(embeddings['twitter'], 'gay', 10)
    most_similar(embeddings['twitter'], 'lesbian', 10)
    most_similar(embeddings['twitter'], 'straight', 10)

def check_sexual_orientation_bias(embeddings):
    get_so_similar_words(embeddings)

def get_ethnic_similarities(embeddings):
    attribute_words = ["intelligent", "criminal", "kind", "violent", "hardworking", "lazy"]
    
    # Wiki Glove Bias check for Ethnic Bias
    black_sim_wiki, white_sim_wiki = word_similarity(embeddings['wiki'], "black", "white", attribute_words)

    # Twitter Glove Bias check for Ethnic Bias
    black_sim_twitter, white_sim_twitter = word_similarity(embeddings['twitter'], "black", "white", attribute_words)

    print('Wiki bias (Ethnic using Adjectives):')
    for i in range(len(attribute_words)):
        print(f'{attribute_words[i]}:\n Black - {black_sim_wiki[i]:.2f} White - {white_sim_wiki[i]:.2f} - higher: {"Black" if black_sim_wiki[i] > white_sim_wiki[i] else "White"}')

    print('Twitter bias (Ethnic using Adjectives):')
    for i in range(len(attribute_words)):
        print(f'{attribute_words[i]}:\n Black - {black_sim_twitter[i]:.2f} White - {white_sim_twitter[i]:.2f} - higher: {"Black" if black_sim_twitter[i] > white_sim_twitter[i] else "White"}')


def check_ethnic_bias(embeddings):
    get_ethnic_similarities(embeddings)


def get_ageism_similarities(embeddings):
    attribute_words = ["smart", "slow", "active", "frail", "energetic", "lazy"]
    
    # Wiki Glove Bias check for Ageism Bias
    young_sim_wiki, old_sim_wiki = word_similarity(embeddings['wiki'], "young", "old", attribute_words)

    # Twitter Glove Bias check for Ageism Bias
    young_sim_twitter, old_sim_twitter = word_similarity(embeddings['twitter'], "young", "old", attribute_words)

    print('Wiki bias (Ageism using Adjectives):')
    for i in range(len(attribute_words)):
        print(f'{attribute_words[i]}:\n Young - {young_sim_wiki[i]:.2f} Old - {old_sim_wiki[i]:.2f} - higher: {"Young" if young_sim_wiki[i] > old_sim_wiki[i] else "Old"}')

    print('Twitter bias (Ageism using Adjectives):')
    for i in range(len(attribute_words)):
        print(f'{attribute_words[i]}:\n Young - {young_sim_twitter[i]:.2f} Old - {old_sim_twitter[i]:.2f} - higher: {"Young" if young_sim_twitter[i] > old_sim_twitter[i] else "Old"}')

def check_ageism_bias(embeddings):
    get_ageism_similarities(embeddings)

def visualize_embeddings(words, embeddings, dataset_name):
    """
    Visualize word embeddings using PCA for dimensionality reduction and Matplotlib for plotting.
    """
    # Extract embeddings for words found in the embeddings dictionary
    valid_words = []
    vectors = []
    
    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])
            valid_words.append(word)
        else:
            print(f"Warning: '{word}' not found in the {dataset_name} embeddings")
    
    # If no valid words were found, exit early
    if len(vectors) == 0:
        print(f"No valid words found in the {dataset_name} embeddings.")
        return
    
    # Convert to numpy array for PCA
    vectors = np.array(vectors)

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plot results using Matplotlib
    plt.figure(figsize=(10, 7))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.01, reduced_vectors[i, 1] + 0.01, word, fontsize=12)

    plt.title(f"PCA Visualization of Word Embeddings ({dataset_name})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

import numpy as np

def visualize_sexual_orientation_bias(embeddings):
    # Words related to sexual orientation
    orientation_words = ["gay", "lesbian", "straight", "homosexual", "heterosexual"]
    
    # Adjectives to check bias
    attribute_words = ["kind", "violent", "hardworking", "lazy", "intelligent", "stupid"]
    
    # Combine orientation words and attribute words
    all_words = orientation_words + attribute_words

    # Check for both Wiki and Twitter embeddings
    for source in ['wiki', 'twitter']:
        print(f"\nVisualizing {source.capitalize()} Embeddings:")
        
        # Filter out words not in the embeddings
        valid_words = [word for word in all_words if word in embeddings.get(source, {})]
        
        if len(valid_words) < len(all_words):
            missing_words = set(all_words) - set(valid_words)
            print(f"Warning: The following words were not found in the {source} embeddings and will be skipped: {missing_words}")
        
        if len(valid_words) == 0:
            print(f"No valid words found in {source} embeddings. Skipping visualization for {source}.")
            continue
        
        # Extract embeddings for valid words
        word_vectors = [embeddings[source][word] for word in valid_words]
        
        # Convert to NumPy array (ensure same dimensionality across words)
        word_vectors = np.array(word_vectors)
        
        if word_vectors.ndim != 2 or word_vectors.shape[1] == 0:
            print(f"Invalid embedding dimensions for {source} embeddings.")
            continue

        # Reduce dimensions to 2D using t-SNE
        # Set perplexity to min(30, len(valid_words) - 1)
        perplexity = min(30, len(valid_words) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        reduced_embeddings = tsne.fit_transform(word_vectors)

        # Create a dataframe for Plotly visualization
        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        df['word'] = valid_words
        df['category'] = ['Orientation' if word in orientation_words else 'Attribute' for word in valid_words]

        # Create interactive scatter plot
        fig = px.scatter(df, x='x', y='y', text='word', color='category', 
                         title=f"{source.capitalize()} Sexual Orientation Bias Visualization")
        fig.update_traces(textposition='top center')
        fig.show()

# Example usage
# visualize_sexual_orientation_bias(embeddings['wiki'])  # Or use any other embedding like 'twitter'


# Main function to compare biases
def main():
    # save_embeddings()
    embeddings = load_saved_embeddings()
    check_gender_bias(embeddings)
    # check_sexual_orientation_bias(embeddings)
    # check_ethnic_bias(embeddings)
    # check_ageism_bias(embeddings)

    # visualizing
    # orientation_words = ["gay", "lesbian", "straight", "homosexual", "heterosexual"]
    # attribute_words = ["kind", "violent", "hardworking", "lazy", "intelligent", "stupid"]
    # words_to_visualize = orientation_words + attribute_words

    # visualize_embeddings(words_to_visualize, embeddings['wiki'], "Wiki")
    # visualize_embeddings(words_to_visualize, embeddings['twitter'], "Twitter")


if __name__ == "__main__":
    main()
