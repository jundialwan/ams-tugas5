import csv

csv.field_size_limit(500000)

#fungsi untuk load dataset
def load_data(dataset):
    sentences = []
    labels = []
    
    with open(dataset, 'rU') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
               text = row['text']
               type = row['type']

               sentences.append(text)
               labels.append(type)
            except:
                continue
    return sentences, labels


##how to load dataset

train_sentences, train_labels = load_data("fake_train.csv")
test_sentences, test_labels = load_data("fake_test.csv")


## training data
#hate 226
#satire 89
#junksci 85
#state 100
#bias 418
#conspiracy 408

## testing data
#hate 20
#satire 11
#junksci 17
#state 21
#bias 22
#conspiracy 22
