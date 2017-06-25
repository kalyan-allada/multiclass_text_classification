# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

#Read train and test data
train_df = pd.read_csv('train.txt',delimiter='\t')
test_df = pd.read_csv('test.txt',header = None ,delimiter="\n")

#Naming the columns in train and test set
train_df.columns = ["label","text"]
test_df.columns = ["text"]

#We will use Porter Stemmer
stemmer = PorterStemmer()

# funtion to implement stemmer
def stem_tokens(tokens, stemmer):
    stemmed_text = []
    for item in tokens:
        stemmed_text.append(stemmer.stem(item))
    return stemmed_text

#function to tokenize text elements and removing numbers and punctuation
def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = text.split(" ")
    stems = stem_tokens(tokens, stemmer)
    return stems

# convert words to their base forms (lemmas) 
def split_into_lemmas(text):
    text = unicode(text, 'utf8').lower()
    words = TextBlob(text).words
    return [word.lemma for word in words]

# Tokenizing and vectorizing the text elements, eleminating stop words, using maximum of 1100 text tokens per document.
vectorizer = TfidfVectorizer(analyzer='word',\
                             tokenizer=tokenize,\
                             ngram_range=(1,3),\
                             lowercase=True,\
                             stop_words ='english',\
                             max_features =1100)
vectorized_features = vectorizer.fit_transform(train_df.text.tolist() + test_df.text.tolist())

#Convert the document term matrix to numpy nd array
vectorized_features_nd = (vectorized_features.toarray())
#print vectorized_features_nd.shape

# Lets try different models 
clf = LinearSVC(penalty = 'l2',dual = True,C=1.0,loss='hinge')
#clf = KNeighborsClassifier()
#clf = MultinomialNB()
#clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, max_features='sqrt')

X_train = vectorized_features_nd[0:len(train_df)]
Y_train = train_df.label
X_test = vectorized_features_nd[len(train_df):]

# split in to train and test set
txt_train, txt_valid, label_train, label_valid = \
    train_test_split(X_train, Y_train, test_size=0.2, random_state=5)

print len(txt_train), len(txt_valid), len(txt_train) + len(txt_valid)

# cross validation
#scores = cross_val_score(clf, txt_train, label_train, cv=5, scoring='accuracy', n_jobs=-1)
#print scores
#print scores.mean(), scores.std()

# Fit model to train subset and predict on validation set
clf = clf.fit(txt_train,label_train)
pred_valid = clf.predict(txt_valid)
print 'accuracy of validation set: ', accuracy_score(label_valid, pred_valid)

label_test = []

# read true labels from file
foput = open("test_labels.txt","r")
for m in foput :
    m = str(m).strip()
    label_test.append(int(m))

# fit entire training set and make prediction on test set
my_model = my_model.fit(X_train,Y_train)
pred_test = my_model.predict(X_test)
print 'accuracy of test set: ', accuracy_score(label_test, pred_test)


