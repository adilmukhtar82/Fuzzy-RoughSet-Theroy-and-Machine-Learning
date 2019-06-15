#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import re  
import nltk
from gensim.models import Word2Vec
import os
from random import seed
from random import randrange
from nltk.corpus import stopwords
import sys
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
# Personal comments: deEmojify removes the emos but it (emos) might have high correlation with the sentiment and emotions
# averaged kth Dimensional M word to reconstruct n tweeet where each tweet has l number of words, total tweets: N
# X: NxM matrix where M is dimensional space (vocab) N number of tweets, X[c][k] represents averaged coefficients for word kth of tweet c


# In[68]:


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def R_a_XY(X, x, y):
    return 1-(abs(x-y)/abs(np.max(X, axis=0)-np.min(X,axis=0)))


def R_XY(X, x, y):
    return min(R_a_XY(X, x, y))

def A(C,y_class):
    if C == y_class:
        return 1
    else:
        return 0
    
def approximations(X, x, y, y_class, C, upper=False):
    if upper:
        return min(R_XY(X, x, y), A(C, y_class))
    else:
        return max(R_XY(X, x, y), A(C, y_class))

def cosine_similarity_approximation(x, y, y_class, C, upper=False):
    if upper:
        #print(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0])
        #print('Min:',min(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0], A(C, y_class)), A(C, y_class), C, y_class)
        return min(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0], A(C, y_class))
    else:
        #print('Max:',max(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0], A(C, y_class)), A(C, y_class), C, y_class)
        return max(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0], A(C, y_class))

def FRNN(X_train, y, y_class, decision_classes):
    ta_ = 0
    class_ = 0
    for C in set(list(decision_classes)):
        ind_where_class_c = np.where(decision_classes == C)
        #print(ind_where_class_c)
        
        #print(X.shape, )
        for ind in ind_where_class_c[0]:
            #print('ind:', ind)
            lower_approx = approximations(X_train, X_train[ind,:], y, y_class, C)
            upper_approx = approximations(X_train, X_train[ind,:], y, y_class, C, upper=True)
            if (lower_approx+upper_approx)/2 >= ta_:
                class_ = C
                ta_ = (lower_approx+upper_approx)/2
    return class_

def FRNN_cosine_sim(X_train, y, y_class, decision_classes):
    ta_ = 0
    class_ = 0
    for C in set(list(decision_classes)):
        #print(C)
        ind_where_class_c = np.where(decision_classes == C)
        #print(ind_where_class_c)
        
        #print(X.shape, )
        for ind in ind_where_class_c[0]:
            #print('ind:', ind)
            lower_approx = cosine_similarity_approximation(X_train[ind,:], y, y_class, C)
            upper_approx = cosine_similarity_approximation(X_train[ind,:], y, y_class, C, upper=True)
            if (lower_approx+upper_approx)/2 >= ta_:
                class_ = C
                ta_ = (lower_approx+upper_approx)/2
    return class_

def cross_validation_split(dataset, folds=5):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

from numpy import linalg as LA
def fuzzy_nearest_neighbour(X, x, y):
    dist = (np.linalg.norm(y-x))**-2/(2-1)
    sum_ = 0
    for vec in X:
        sum_ += (np.linalg.norm(y-vec))**-2/(2-1)
    print(sum_, dist)
    return dist/sum_


# In[69]:


if __name__ == '__main__':

    df = pd.read_csv(sys.argv[1])
    #df = pd.read_csv('data/anger_detection_data.csv')
    stop_words = set(stopwords.words('english'))

    tweets = list()
    for t in df['Tweet']:
        t = ' '.join([word for word in t.split() if word.lower() not in stop_words])
        tweets.append(str(deEmojify(t).lower()))

    tokenized = [sentence.split() for sentence in tweets]

    #print(int(round(len(model.wv.vocab)/1.5)), len(model.wv.vocab))
    model = Word2Vec(tokenized, min_count=5, size = 300, window=5, negative = 10, workers = multiprocessing.cpu_count())
    X = np.zeros((len(df), len(model.wv.vocab)))
    y = np.zeros((len(df), 1))
    unknown_words = 0
    for c, tokens in enumerate(tokenized):
        for k, word in enumerate(tokens):
            try:
                X[c][k] = (np.average(model[word])+X[c][k])/2
            except:
                unknown_words+=1
                continue
        y[c][0] = int(df['Intensity Class'][c].split(":")[0])
    #print('Percentage of unknown words:', unknown_words/len(model.wv.vocab))
    folds = cross_validation_split(np.append(X, y, axis = 1))
    frnn_acc = list()
    for fold_counter, f in enumerate(folds):
        data_matrix = np.asarray(f)
        decision_classes = data_matrix[:,len(model.wv.vocab)]
        X = np.delete(data_matrix, len(model.wv.vocab), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, decision_classes, test_size = 0.40)
        predictions = np.zeros((len(X_test), 1))
        
        for c, y in enumerate(X_test):
            predictions[c][0] = FRNN(X_train, y, y_test[c], y_train)
            #predictions[c][0] = FRNN_cosine_sim(X_train, y, y_test[c], y_train)
            
        frnn_acc.append(accuracy_score(y_test, predictions))
        print('Fold %d Accuracy %.2f FRNN '%(fold_counter+1, accuracy_score(y_test, predictions)))
    print('Average Accuracy %.2f FRNN'%np.average(frnn_acc))
    print('Std %.2f FRNN'%np.std(frnn_acc))





