import nltk
import random
from nltk.corpus import udhr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('udhr')

# vectorizer = CountVectorizer(ngram_range=(1,3), analyzer='char')

# # count all unigrams, bigrams. and trigrams in the sentences and create a feature vector
# sentences = ["It was the best of times, it was the worst of times,",
#              "It was the age of wisdom, it was the age of foolishness,"]

# vector = vectorizer.fit_transform(sentences)

# print("Unigrams, bigrams, and trigrams:", vectorizer.get_feature_names_out())
# print("\nFeatures:")
# print(vector.toarray())

def train_udhr(pipeline):
    X = []
    y = []

    # The UDHR is quite small, so let's create 1,000 "fake" sentences in each
    # language by randomly stringing together 3-15 words
    for lang in languages:
        words = udhr.words(lang)
        sents = [" ".join(random.choices(words, k=random.randint(3,15))) for x in range(1000)]
        X.extend(sents)
        y += [lang] * len(sents)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)

    pipeline.fit(X_train, y_train)
    return X_test, y_test

languages = ['Icelandic_Yslenska-Latin1',
             'Norwegian-Latin1',
             'Swedish_Svenska-Latin1',
             'Danish_Dansk-Latin1',
             'Finnish_Suomi-Latin1',
             'Faroese-Latin1']

# pipeline with CountVectorizer of char level 
# n-grams generating unigram, bigram, and trigram counts
# using logistic regression classifier and liblinear solver
q1_pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3), analyzer='char')),
    ('clf', LogisticRegression(solver='liblinear'))
])


X_test, y_test = train_udhr(pipeline=q1_pipeline)
score = q1_pipeline.score(X_test, y_test)
print("Accuracy: {:.1%}".format(score))

example_icelandic = "Þykjustustríðið var átta mánaða tímabil í upphafi seinni heimsstyrjaldarinnar þar sem allt var með kyrrum kjörum á vesturvígsstöðvum. Aðeins ein hernaðaraðgerð átti sér stað þegar Frakkar gerðu tilraun til sóknar í Saarland. Átök styrjaldaraðila voru að mestu bundin við sjóinn. Þetta tímabil stóð yfir þar til orrustan um Frakkland hófst þann 10. maí 1940.".split('.')[:-1]
example_norwegian = "Området hadde vært bebodd av urbefolkningen i lang tid før europeerne kom dit, og regionen har noen av de tidligste bevisene på menneskelig bosetning i Nord-Amerika. Urbefolkningen i Yukon hadde svært omfattende handelsnettverk. . Klondike-gullrushet, som begynte i 1896, førte til en tilstrømming av anslagsvis 100 000 gullgravere til den tynt befolkede regionen.".split('.')[:-1]
example_swedish = "Praktblåsmyg (Malurus splendens) är en liten långstjärtad tätting i familjen blåsmygar (Maluridae). Den förekommer över stora delar av den australiska kontinenten från centrala västra New South Wales och sydvästra Queensland till Western Australias kust och bebor främst torra eller semitorra regioner.".split('.')[:-1]
example_danish = "Simon Kvamm (født 1975) er en dansk kunstner. Han er musiker og forsanger i bandet Hugorm, som blev til i Klitmøller i 2017 i samarbejde med Arní Bergmann og Morten Gorm. Bandets første EP kom i 2020 med navnet Folk skal bare holde deres kæft, og senere samme år udkom albummet Kom vi flygter. I oktober 2022 udkom bandets andet album Tro, Hug & Kærlighed.".split('.')[:-1]
example_finnish = "Pietarsaaren kaupungissa asuu noin 19 000 henkilöä ja laajemmin koko seudulla asuu noin 50 000 henkilöä. Asukasluvultaan Pietarsaari on Suomen 60:nneksi suurin kaupunki. Pietarsaari on kaksikielinen: 54 prosenttia asukkaista puhui äidinkielenään ruotsia ja 31 prosenttia suomea.".split('.')[:-1]
example_faroese = "Tann 64 ára gamli norski høvundin Jon Fosse (myndin) fær heiðursløn Nobels í bókmentum fyri 2023. Petteri Orpo er nýggjur forsætisráðharri í Finnlandi. Umframt Savningarflokkin eru tjóðskaparligi flokkurin Sannir Finnar eins og Svensk Folkeparti og Kristindemokratarnir við í stjórnarsamgonguni.".split('.')[:-1]

example_sentences = []
example_sentences.extend(example_icelandic)
example_sentences.extend(example_norwegian)
example_sentences.extend(example_swedish)
example_sentences.extend(example_danish)
example_sentences.extend(example_finnish)
example_sentences.extend(example_faroese)


predictions = q1_pipeline.predict(example_sentences)

# Print out the predictions
print("Predicted languages for the example sentences:")
for sentence, prediction in zip(example_sentences, predictions):
    print(f"Sentence: '{sentence}' -> Predicted language: {prediction}")

# seems to do great at Icelandic,but the others it is definitely not at 95% accuracy