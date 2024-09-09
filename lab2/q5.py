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
top200 = [token for (token, count) in nltk.FreqDist(training_tokens).most_common(200)]
print(top200)