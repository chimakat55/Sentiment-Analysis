# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:07:30 2022

@author: Manisha 
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

stops = set(stopwords.words('english')+list(string.punctuation))

datatrain = pd.read_csv('sentiment_train.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

datatrain.head()

######### Pre-processing training data ##########
# lowercase, punctuation and stop words removed #

datatrain['text']=[string.lower() for string in datatrain['text']]
datatrain['cleantext']=datatrain['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
datatrain['tokens']=[string.split(' ') for string in datatrain['cleantext']]
datatrain['vocab']=[set(row) for row in datatrain['tokens']]


## Tf idf Vectorization ##

tfidf = TfidfVectorizer(max_features=5000)
x = datatrain['text']
y = datatrain['sentiment']

x = tfidf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = LinearSVC()
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))


