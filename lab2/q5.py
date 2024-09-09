import nltk
import random 

# question 5 - class of only emotions and whQuestions

nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()
emotAndWhQuestions = [post for post in posts if post.get("class") == "Emotion" or post.get("class") == "whQuestion"]

# q5 create test and training datasets

random.shuffle(emotAndWhQuestions)
chat_training = emotAndWhQuestions[:1300]
chat_testing = emotAndWhQuestions[1300:]

training_tokens = []
for post in chat_training:
    tokens = nltk.word_tokenize(post.text)
    training_tokens.extend(tokens)

# print(tokens[:500])

training_fd = nltk.FreqDist(training_tokens)
top200 = [token for (token, count) in training_fd.most_common(200)]
# print(top200)


def get_word_features(inputString):
    return {word: word in inputString for word in top200}

customFeatures = ["who", "what", "when", "where", "why", "how", "!", "?", "lol", "hahaha", "haha", ":)", ":(", "lmao", "rofl", "teehee", "XD", ":-)", "8D", ":X", ":-/", ":/", "<3", ":'(","love", "hate", "like", "dislike", "sad", "happy", "angry", "mad", "annoyed", "excited", "bored", "scared", "fear", "afraid", "surprised", "surprise", "disgusted", "disgust", "shocked", "shock", "confused", "confuse", "confusing", "confusion", "depressed", "depress", "depressing", "depression", "anxious", "anxiety", "anxious", "anxiously","damn", "omg", ";-)", "<3", "grrr", "hehehe", "hehe", ":p", ":P",":)", ":(", ":D", ":P", ":/", ":|", ":O", ":S", ":*", ":'(","haha", "hahaha", "lol", "lmao", "rofl", "O:", ":O", "o:", ":o", "wtf"]

def get_custom_features(inputString):
    return {word: word in inputString for word in customFeatures}

# q5 training and output

# create training sets and train classifiers
word_training = [(get_word_features(post.text), post.get("class")) for post in chat_training]
custom_training = [(get_custom_features(post.text), post.get("class")) for post in chat_training]
word_classifier = nltk.NaiveBayesClassifier.train(word_training)
custom_classifier = nltk.NaiveBayesClassifier.train(custom_training)

# create testing sets and test classifiers
word_test = [(get_word_features(post.text), post.get("class")) for post in chat_testing]
custom_test = [(get_custom_features(post.text), post.get("class")) for post in chat_testing]

print("Word accuracy:", nltk.classify.accuracy(word_classifier, word_test))
print("Custom accuracy:", nltk.classify.accuracy(custom_classifier, custom_test))

word_classifier.show_most_informative_features(20)
custom_classifier.show_most_informative_features(20)