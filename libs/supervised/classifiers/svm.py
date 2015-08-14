#!/usr/bin/python
# -*- coding: utf-8 -*-

from libs.supervised.supervised_interface import SupervisedInterface
from util.cache import simple_caching
from util.common import hash_obj

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer

# from nltk.stem.porter import PorterStemmer

import re
from string import punctuation

STOPWORDS_PATH = 'data/stopwords.txt'

citations1 = re.compile("\\(\\s?(((\\w+)((\\sand\\s\\w+)|(\\set\\sal\\.))?"
                        ",?\\s\\d+)(\\sand\\s|[,;]\\s))*((\\w+)((\\sand\\s"
                        "\\w+)|(\\set\\sal\\.))?,?\\s\\d+)\\)")
citations2 = re.compile("\\[(\\d+,?)+\\](, )?")
tables = re.compile("\\(\\s?((\\w+[–\\-,;:\\./]?\\s)+)?([Tt]ab(les?|"
                    "\\.))\\s((\\w+[–\\-,;:\\./]*\\s?)+)?\\)")
figures = re.compile("\\(\\s?((\\w+[–\\-,;:\\./]?\\s)+)?([Ff]ig(ures?"
                     "|\\.))\\s((\\w+[–\\-,;:\\./]*\\s?)+)?\\)")
numbers = re.compile('(\\s|^)\\S*\\d+\\S*(\\s|$)')


class Supervised(SupervisedInterface):

    supervised_opts = {'skip-tokenization': {'action': 'store_true',
                                             'default': False}}

    def __init__(self, args, opts):
        super(Supervised, self).__init__(args, opts)
        with file(STOPWORDS_PATH) as f:
            self.stopwords = frozenset([l.strip().lower() for l in f])
        # self.stemmer = PorterStemmer()
        self.punctuation = set(punctuation)
        if hasattr(opts, 'cachedir'):
            self.cachedir = opts.cachedir
        else:
            self.cachedir = 'cache'

    def clean(self, txt):
        return self._clean_svm_input(txt, cache_comment=hash_obj(txt))

    @simple_caching()
    def _clean_svm_input(self, txt):
        txt = tables.sub('', txt)
        txt = figures.sub('', txt)
        txt = citations1.sub('', txt)
        txt = citations2.sub('', txt)
        txt = txt.replace('-', ' ')
        txt = numbers.sub(' ', txt)
        txt = ''.join([t for t in txt if t not in self.punctuation])
        txt = txt.lower()
        # terms = [t.lower() for t in txt.split()
        #          if t.lower() not in self.stopwords]

        # terms = [self.stemmer.stem(st) for t in terms]
        # txt = ' '.join(terms)
        return txt

    def train(self, X_train, y_train):
        svc_params = {'class_weight': 'auto', 'kernel': 'linear'}
        if getattr(self.opts, 'probability', False):
            svc_params['probability'] = True
        if self.opts.skip_tokenization:
            self.pl = Pipeline([('clf', SVC(**svc_params))])
        else:
            cnt_vect_para = {'ngram_range': (1, 3), 'preprocessor': self.clean}
            self.pl = Pipeline([('vect', CountVectorizer(**cnt_vect_para)),
                                ('tfidf', TfidfTransformer()),
                                ('clf', SVC(**svc_params))])
        self.pl.fit(X_train, y_train)

    def run(self, X_test):
        if getattr(self.opts, 'probability', False):
            return self.pl.predict_proba(X_test)
        else:
            return self.pl.predict(X_test)
