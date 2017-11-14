#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import IPython as ipy
from toolz import compose
from sklearn.naive_bayes import GaussianNB

APP_TWEETS_CSV = pd.read_csv('./app_tweets.csv', quotechar='`')
OTHER_TWEETS_CSV = pd.read_csv('./other_tweets.csv', quotechar='`')

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

def count(tokens):
    d = {}
    for token in tokens:
        d[token] = d.get(token, 0) + 1
    return d


def main():
    f = compose(tokenize, trim)
    A = count(f(t) for t in APP_TWEETS_CSV['tweet'])
    B = count(f(t) for t in OTHER_TWEETS_CSV['tweet'])


if __name__ == '__main__':
    main()
