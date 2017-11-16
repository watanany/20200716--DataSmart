#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from operator import add
from functools import reduce
from toolz import compose
from sklearn.naive_bayes import GaussianNB

import IPython as ipy

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

def count(tokens):
    d = {}
    for token in tokens:
        d[token] = d.get(token, 0) + 1
    return d

def create_feature_vector(tokens, all_tokens):
    C = count(tokens)
    return np.array([C.get(t, 0) for t in all_tokens])

def split_list(L, n):
    result = []
    for i in range(len(L)):
        if i % (len(L) / n) != 0:
            result[-1].append(L[i])
        else:
            result.append([])
            result[-1].append(L[i])
    return result

def main():
    A = set(reduce(add, [preprocess(t) for t in APP_TWEETS_CSV['tweet']]))
    O = set(reduce(add, [preprocess(t) for t in OTHER_TWEETS_CSV['tweet']]))
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
    print('label: {}'.format(g.predict(T_A)))
    print('label: {}'.format(g.predict(T_O)))


if __name__ == '__main__':
    main()
