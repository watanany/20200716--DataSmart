#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from operator import add
from functools import reduce
from toolz import compose
from sklearn.naive_bayes import GaussianNB

APP_TWEETS_CSV = pd.read_csv('./app_tweets.csv', quotechar='`')
OTHER_TWEETS_CSV = pd.read_csv('./other_tweets.csv', quotechar='`')
LABEL_APP = 0
LABEL_OTHER = 1

def trim(text):
    text = text.lower()
    text = text.replace('. ', '')
    text = text.replace(': ', '')
    text = text.replace('?', '')
    text = text.replace('!', '')
    text = text.replace(';', '')
    text = text.replace(',', '')
    return text

def tokenize(text):
    return text.split(' ')

def preprocess(tweet):
    return compose(tokenize, trim)(tweet)

def create_feature_vector(tokens, all_tokens):
    return np.array([tokens.count(t) for t in all_tokens])

def split_list(L, n):
    indices= [0] + [len(L) // i for i in range(n, 0, -1)]
    return [L[i0:i1] for i0, i1 in zip(indices[0:], indices[1:])]

def calc_accuracy(labels, answer):
    a = len([label for label in labels if label == answer])
    b = len(labels)
    return a / b

def main():
    A = { *reduce(add, [preprocess(t) for t in APP_TWEETS_CSV['tweet']]) }
    O = { *reduce(add, [preprocess(t) for t in OTHER_TWEETS_CSV['tweet']]) }
    all_tokens = sorted(A | O)

    app_tweets_list = split_list(list(APP_TWEETS_CSV['tweet']), 2)
    other_tweets_list = split_list(list(OTHER_TWEETS_CSV['tweet']), 2)

    X = [create_feature_vector(preprocess(tweet), all_tokens) for tweet in app_tweets_list[0] + other_tweets_list[0]]
    y = [LABEL_APP] * len(app_tweets_list[0]) + [LABEL_OTHER] * len(other_tweets_list[0])
    T_A = [create_feature_vector(preprocess(tweet), all_tokens) for tweet in app_tweets_list[1]]
    T_O = [create_feature_vector(preprocess(tweet), all_tokens) for tweet in other_tweets_list[1]]

    X = np.array(X)
    y = np.array(y)
    T_A = np.array(T_A)
    T_O = np.array(T_O)

    g = GaussianNB()
    g.fit(X, y)
    print('accuracy(app):\t{:.1f}%'.format(calc_accuracy(g.predict(T_A), LABEL_APP) * 100.0))
    print('accuracy(other):\t{:.1f}%'.format(calc_accuracy(g.predict(T_O), LABEL_OTHER) * 100.0))


if __name__ == '__main__':
    main()
