# -*- coding: utf-8 -*-

## Authors: Audrey, Mamen and Manisha ##

# INSTRUCTIONS: You are responsible for making sure that this script outputs 

# 1) the evaluation scores of your system on the data in CSV_TEST (minimally 
# accuracy, if possible also recall and precision).

# 2) a csv file with the contents of a dataframe built from CSV_TEST that 
# contains 3 columns: the gold labels, your system's predictions, and the texts
# of the reviews.

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from random import sample
import nltk
from nltk.corpus import stopwords
import string
from sklearn.pipeline import Pipeline

stops = set(stopwords.words('english')+list(string.punctuation))


TRIAL = 0

# ATTENTION! the only change that we are supposed to do to your code
# after submission is to change 'True' to 'False' in the following line:
EVALUATE_ON_DUMMY = True

# the real thing:
CSV_TRAIN = pd.read_csv('sentiment_train.csv')
CSV_VAL = pd.read_csv('sentiment_val.csv')
CSV_TEST = pd.read_csv('sentiment_dummy_test_set.csv') #you dont have this file; we do

if TRIAL:
    CSV_TRAIN = pd.read_csv('sentiment_train.csv')
    CSV_VAL = pd.read_csv('sentiment_val.csv')
    CSV_TEST = pd.read_csv('sentiment_dummy_test_set.csv')
    print('You are using your SMALL dataset!\n')
elif EVALUATE_ON_DUMMY:
    CSV_TEST = pd.read_csv('sentiment_dummy_test_set.csv')
    print('You are using the FULL dataset, and using dummy test data! (Ok for system development.)')
else:
    print('You are using the FULL dataset, and testing on the real test data.')
    
    
#below here is my code not theirs
datatrain = pd.read_csv('sentiment_train.csv') 
dataval= pd.read_csv('sentiment_val.csv')
CSV_TEST = pd.read_csv('sentiment_dummy_test_set.csv')


######### Pre-processing training data ##########
# lowercase, punctuation and stop words removed #

datatrain['text']=[string.lower() for string in datatrain['text']]
datatrain['cleantext']=datatrain['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
datatrain['tokens']=[string.split(' ') for string in datatrain['cleantext']]
datatrain['vocab']=[set(row) for row in datatrain['tokens']]

########## Inserting Positive and Negative Word Lists and a complete Sentiment Word List#########

with open('positive-words.txt') as file:
    pos_words = file.readlines()
    pos_words = [line.rstrip() for line in pos_words]
    
with open('negative-words.txt') as file:
    neg_words = file.readlines()
    neg_words = [line.rstrip() for line in neg_words]
    
sent_words = pos_words + neg_words
    
########## Features ##########
# Feature one is number of positive words #
datatrain['n_pos_words'] = [np.sum([row.count(positive_word) for positive_word \
                             in pos_words]) for row in datatrain['tokens']]
# Feature two is number of negative words#
datatrain['n_neg_words']=[np.sum([row.count(negative_word) for negative_word \
                              in neg_words]) for row in datatrain['tokens']]
# Feature three is lexical diversity #
datatrain['lex_diversity']=[len(row)for row in datatrain['vocab']]

# Feature four is excluding positive words that come (2 places) after "not" from being classified as positive sentiment #
i=0 
pos = []
for row in datatrain['tokens']:
    pos.append(0)
    for word in pos_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0: 
                if not 'not' == row[position -2]:
                    pos[i]=pos[i] + row.count(word)
    i=i+1
datatrain['n_pos_no_not']=pos

# Feature five is excluding negative words that come (2 places) after "not" from being classified as negative sentiment #
i=0 
neg = []
for row in datatrain['tokens']:
    neg.append(0)
    for word in neg_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0:
                if not 'not' == row[position -2]:
                    neg[i]=neg[i] + row.count(word)
    i=i+1
datatrain['n_neg_no_not']=neg


# feature six is looking only at the last sentiment word in the line to determine whether that is positive or negative
positive_negative = []
positive_negative_list =[]
for row in datatrain['tokens']:
    positive_negative = []
    for word in sent_words:
        if word in row:
            positive_negative=word
    positive_negative_list.append(positive_negative)

datatrain['final_sent_word']= positive_negative_list

final_sent=[]
for row in datatrain['final_sent_word']:
    finalsent=0
    for word in pos_words:
        if word in row:
            finalsent=1
    final_sent.append(finalsent)
datatrain['final_sent']=final_sent  

######### Vectorization using tfidf #########

## tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
x_train = datatrain[['n_pos_words', 'n_neg_words','n_pos_no_not','n_neg_no_not','final_sent','lex_diversity']]
y_train = datatrain['sentiment']

## x_train = tfidf.fit_transform(x_train)   

## lr = LogisticRegression()
## tfidf_lr_pipe = Pipeline([('tfidf', tfidf), ('lr', lr)])

## tfidf_lr_pipe.fit(x_train, y_train)  

## algorithm - logistic regression ## 

model = LogisticRegression()
X = datatrain[['n_pos_words', 'n_neg_words','n_pos_no_not','n_neg_no_not','final_sent','lex_diversity']]
y = datatrain['sentiment']
model = model.fit(X, y)
datatrain['predicted_by_logistic_regression'] = model.predict(X)

######### Training error analysis ##########
print('Training Data Results:')
print(pd.crosstab(datatrain['sentiment'], datatrain['predicted_by_logistic_regression']))
SK_accuracy=accuracy_score(y_train,datatrain['predicted_by_logistic_regression'])
print(f'SK Accuracy linear regression: {SK_accuracy}')
precision=precision_score(y_train,datatrain['predicted_by_logistic_regression'], average='binary', pos_label='pos')
recall=recall_score(y_train,datatrain['predicted_by_logistic_regression'], average='binary', pos_label='pos')
print(f'Precision Linear Regression: {precision}')
print(f'Recall Linear Regression: {recall}\n')

print('Moving to Validation Data:\n')

########## Reiterated with validation data #########

######### Pre-processing training data ##########
# lowercase, punctuation and stop words removed #

dataval['text']=[string.lower() for string in dataval['text']]
dataval['cleantext']=dataval['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stops)]))
dataval['tokens']=[nltk.word_tokenize(string) for string in dataval['cleantext']]
dataval['vocab']=[set(row) for row in dataval['tokens']]

# Feature one is positive words list #

dataval['n_pos_words'] = [np.sum([row.count(positive_word) for positive_word \
                               in pos_words]) for row in dataval['tokens']]
    
# Feature two is negative words list #

dataval['n_neg_words']=[np.sum([row.count(negative_word) for negative_word \
                                  in neg_words]) for row in dataval['tokens']]

# Feature three is positive words list #

dataval['lex_diversity']=[len(row)for row in dataval['vocab']]

# Feature five is excluding negative words that come after "not" from being classified as negative sentiment #
i=0 
neg = []
for row in dataval['tokens']:
    neg.append(0)
    for word in neg_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0:
                if not 'not' == row[position -2]:
                    neg[i]=neg[i] + row.count(word)
    i=i+1
dataval['n_neg_no_not']=neg

# Feature four is excluding positive words that come after "not" from being classified as positive sentiment #

i=0 
pos = []
for row in dataval['tokens']:
    pos.append(0)
    for word in pos_words: 
        if row.count(word) > 0:
            position= row.index(word)
            if (position -2) >= 0: 
                if not 'not' == row[position -2]:
                    pos[i]=pos[i] + row.count(word)
    i=i+1
dataval['n_pos_no_not']=pos 

# feature six is looking only at the last sentiment word in the line to determine whether that is positive or negative
positive_negative = []
positive_negative_list =[]
for row in dataval['tokens']:
    positive_negative = []
    for word in sent_words:
        if word in row:
            positive_negative=word
    positive_negative_list.append(positive_negative)

dataval['final_sent_word']= positive_negative_list

final_sent=[]
for row in dataval['final_sent_word']:
    finalsent=0
    for word in pos_words:
        if word in row:
            finalsent=1
    final_sent.append(finalsent)
dataval['final_sent']=final_sent   

######### Vectorization using tfidf #########

## tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
x_val = dataval[['n_pos_words', 'n_neg_words','n_pos_no_not','n_neg_no_not','final_sent','lex_diversity']]
y_val = dataval['sentiment']

## x_val = tfidf.fit_transform(x_val)   

## lr = LogisticRegression()
## tfidf_lr_pipe = Pipeline([('tfidf', tfidf), ('lr', lr)])

## tfidf_lr_pipe.fit(x_val, y_val)  

########## algorithm - logistical regression ##########

X_val = dataval[['n_pos_words', 'n_neg_words','n_pos_no_not','n_neg_no_not','final_sent','lex_diversity']]
y_val = dataval['sentiment']
dataval['predicted_by_logistic_regression'] = model.predict(X_val)

######### Validation error analysis ##########
print('Validation Data Results:')
print(pd.crosstab(dataval['sentiment'], dataval['predicted_by_logistic_regression']))
SK_accuracy=accuracy_score(y_val,dataval['predicted_by_logistic_regression'])
print(f'SK Accuracy linear regression: {SK_accuracy}')
precision=precision_score(y_val,dataval['predicted_by_logistic_regression'], average='binary', pos_label='pos')
recall=recall_score(y_val,dataval['predicted_by_logistic_regression'], average='binary', pos_label='pos')
print(f'Precision Linear Regression: {precision}')
print(f'Recall Linear Regression: {recall}')


# Sample of false negs and pos to help us understand where we're getting problems. To be deleted before submission.  
gold_neg = y_val == 'neg'
pred_pos = dataval['predicted_by_logistic_regression'] == 'pos'

gold_pos = y_val == 'pos'
pred_neg = dataval['predicted_by_logistic_regression'] == 'neg'

dataval['false_positives'] = gold_neg & pred_pos
dataval['false_negatives'] = gold_pos & pred_neg

twenty_false_positives = dataval.sample(n=20, weights='false_positives')
twenty_false_negatives = dataval.sample(n=20, weights='false_negatives')

print('#' * 60)
print('FALSE POSITIVES:')
for example in twenty_false_positives['text']:
    print(example)
print('#' * 60)
print('FALSE NEGATIVES:')
for example in twenty_false_negatives['text']:
    print(example)
