import numpy as np
from data import load_data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB # .61
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# load data from given csv file
train_sentences, train_labels = load_data('fake_train.csv')

# change every sentences in the corpus to TF-IDF vector
# v = TfidfVectorizer(smooth_idf=False)
vectorizer = CountVectorizer(stop_words='english')
train_tfidf_vector = vectorizer.fit_transform(train_sentences)

# begin train the model
# model = LogisticRegression().fit(train_tfidf_vector, train_labels)
# model = KNeighborsClassifier(n_neighbors=5).fit(train_tfidf_vector, train_labels)
model = MultinomialNB().fit(train_tfidf_vector, train_labels)

# load test data from given csv
test_sentences, test_labels = load_data('fake_test.csv')

test_tfidf_vector = vectorizer.transform(test_sentences)

prediction = model.predict(test_tfidf_vector)

accuracy = accuracy_score(test_labels, prediction)

print 'Accuracy: ', accuracy
