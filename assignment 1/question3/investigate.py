import gensim.downloader as api
from gensim.models import KeyedVectors

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

# Check bias by comparing word similarities
def check_bias(model, word1, word2, targets):
    similarities_word1 = [model.similarity(word1, target) for target in targets]
    similarities_word2 = [model.similarity(word2, target) for target in targets]
    return similarities_word1, similarities_word2

# Investigate bias in relationship between gender and profession
def check_gender_bias(embeddings):
    profession_words = ["doctor", "nurse", "scientist", "teacher"]
    # Wiki Glove Bias check
    male_sim_wiki, female_sim_wiki = check_bias(embeddings['wiki'], "man", "woman", profession_words)

    # Twitter Glove Bias check
    male_sim_twitter, female_sim_twitter = check_bias(embeddings['twitter'], "man", "woman", profession_words)

    print('Wiki bias:')
    for i in range(len(profession_words)):
        print(f'{profession_words[i]}:\n m - {male_sim_wiki[i]:.2f} f - {female_sim_wiki[i]:.2f} - higher: {"male" if male_sim_wiki[i] > female_sim_wiki[i] else "female"}')

    print('Twitter bias:')
    for i in range(len(profession_words)):
        print(f'{profession_words[i]}:\n m - {male_sim_twitter[i]:.2f} f - {female_sim_twitter[i]:.2f} - higher: {"male" if male_sim_twitter[i] > female_sim_twitter[i] else "female"}')

# Main function to compare biases
def main():
    embeddings = load_saved_embeddings()
    check_gender_bias(embeddings)


if __name__ == "__main__":
    main()
