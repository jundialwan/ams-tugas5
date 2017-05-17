import re
from data import load_data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from stemming.porter2 import stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB # .61
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# load data from given csv file
train_sentences, train_labels = load_data('fake_train.csv')

cleaned_train_sentences = []

for sentence in train_sentences:
    free_dot_sentence_tokens = word_tokenize(re.sub('[.,]', '', sentence))
    free_stopwords_sentence = []
    stpwrds = set(stopwords.words('english'))

    for word in free_dot_sentence_tokens:
        if word not in stpwrds:
            free_stopwords_sentence.append(word)

    cleaned_train_sentences.append(' '.join(free_stopwords_sentence))

# algoritm used
#   LogisticRegression()
#   KNeighborsClassifier(n_neighbors=3)
#   MultinomialNB()
#   RandomForestClassifier()
#   MLPClassifier()
#   xgb.XGBClassifier()
#	GradientBoostingClassifier()
algoritm = xgb.XGBClassifier()

model = Pipeline([
    ('vectorizing', CountVectorizer(stop_words='english')),
    ('classifying', algoritm)])

model.fit(cleaned_train_sentences, train_labels)

# load test data from given csv
test_sentences, test_labels = load_data('fake_test.csv')

prediction = model.predict(test_sentences)

accuracy = accuracy_score(test_labels, prediction)

print 'Accuracy: ', accuracy
