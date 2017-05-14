import numpy as np
from data import load_data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB # .61
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# load data from given csv file
train_sentences, train_labels = load_data('fake_train.csv')

# change every sentences in the corpus to TF-IDF vector
# v = TfidfVectorizer(smooth_idf=False)
# vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=False)
# train_tfidf_vector = vectorizer.fit_transform(train_sentences)

# begin train the model
# model = LogisticRegression().fit(train_tfidf_vector, train_labels) #.77
# model = KNeighborsClassifier(n_neighbors=3).fit(train_tfidf_vector, train_labels) # .61
# model = MultinomialNB().fit(train_tfidf_vector, train_labels) # .33
# model = RandomForestClassifier().fit(train_tfidf_vector, train_labels) # .62
# model = MLPClassifier().fit(train_tfidf_vector, train_labels) # .70

model = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=None, smooth_idf=False)),
    ('classifier', MultinomialNB())])

model.fit(train_sentences, train_labels)

# load test data from given csv
test_sentences, test_labels = load_data('fake_test.csv')

# test_tfidf_vector = vectorizer.transform(test_sentences)

# prediction = model.predict(test_tfidf_vector)
prediction = model.predict(test_sentences)

accuracy = accuracy_score(test_labels, prediction)

print 'Accuracy: ', accuracy