#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on May 5, 2015

@author: rmn
'''
from argparse import (ArgumentParser, Namespace)
import numpy as np
import numpy.linalg as LA
from _collections import defaultdict
from random import randint
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from heapq import heappush, heappushpop
try:
    import cPickle as pickle
except:
    import pickle
import os
import re
from util.tokenization import WordTokenizer, SentTokenizer
from nltk.stem.porter import PorterStemmer
from util.common import (hash_obj, VerbosePrinter, flatten)
from constants import get_path

cache_DIR = get_path()['cache']


class Summarizer(object):
    '''
    Base class for summarizers
    '''
    method_opts = {}

    def __init__(self, args=None, opts=None):
        """ Initialize the Summarizer.
            args is a list of arguments for the Summarizer (typically
            from input evaluate.py.
            opts is a ArgumentParser or OptionParser object.

            Notes
            ----------
            If you overwrite this method, remember to call super,
            otherwise the options in method_opts will not be parsed.
        """
        if opts:
            self.opts = opts
        else:
            self.opts = Namespace()

        self.args = args
        self._parse_args()

        # attribute error when type(self.opts) is Namespace
        try:
            self.cachedir = getattr(
                self, 'opts', {}).get('cachedir', None)
        except AttributeError:
            self.cachedir = vars(self.opts).get('cachedir', None)

        if not self.cachedir:
            self.cachedir = cache_DIR

        self.hashcomm = hash_obj({k: vars(self.opts)[k]
                                  for k in self.method_opts
                                  if hasattr(opts, k)})

        if not hasattr(self.opts, 'printer'):
            self.printer = VerbosePrinter(getattr(opts, 'verbose', False))

    def clean_citation(self, t):
        '''
        Replaces the author names (citation marker) in the citation
        '''
        regex_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
                                    r"(\[(\d+([,–]\s?)?)+\])|"
                                    r"\[[\d,-]+\]").sub
        return regex_citation('', t)

    def pick_from_cluster(self, cluster, max_length=250, weighted=False, mode=None, lmb=0.3):
        '''
        Picks sentences from a cluster of sentences

        Args:
            cluster(dict) -- A dictionary whose keys are cluster ids
                and values are list of sentences belonging to that cluster
            max_length(int) -- maximum length of the summary (in words)
            weighted(bool) -- Weighted based on the number of sentences in 
                each cluster
        '''
        word_tokenize = WordTokenizer(stem=False)
        final_sum = []
        if not mode:
            if weighted:
                counts = defaultdict(int)
                idx = {}
                for k in cluster:
                    counts[k] += 1
                    idx[k] = 0
                num_sents = max_length / float(50) + 2

                while len(final_sum) < num_sents and\
                        word_tokenize.count_words(final_sum) < max_length:
                    weighted_choice = [(k, v) for k, v in counts.iteritems()]
                    avg_cnt = np.mean(counts.values())

                    def cnvrt(cnt, idx, avg_cnt):
                        if cnt < 2 and idx == 0:
                            return (cnt + avg_cnt) / 2
                        else:
                            return cnt
                    population = [
                        val for val, cnt in weighted_choice
                        for _ in range(cnvrt(cnt, idx[val], int(avg_cnt)))]
                    to_pick = random.choice(population)
                    idx[to_pick] += 1
                    if (idx[to_pick] < len(cluster[to_pick])) and\
                        (word_tokenize.count_words(final_sum) <
                         max_length):
                        final_sum.append(cluster[to_pick][idx[to_pick]])

            else:
                idx = 0
                end = False
                while word_tokenize.count_words(final_sum) < max_length and not end:
                    for k in cluster:
                        if (idx < len(cluster[k])) and\
                                (word_tokenize.count_words(final_sum) <
                                 max_length):
                            final_sum.append(cluster[k][idx])
                    idx += 1
                    if idx > 10:
                        end = True
        elif mode == 'mmr':
            def summarize1(self, doc, max_length=10):
                '''
                Summarizes a document or list of docs
                MMR(S) = lambda*Sim(S,D)-(1-lambda)*Sim(S,Summary)
                Arg:
                    doc: (list) | (str)

                    max_length: The maximum length of the desired summary

                Returns
                -------
                str
                '''
                if isinstance(doc, str):  # list of sentences, no need to tokenize
                    s_t = SentTokenizer()
                    docs = s_t(doc, offsets=False)
                    docs += [doc]  # Dummy sentence, The whole document
                else:
                    docs = doc + [' '.join(doc)]
                tokzr = self.get_tokenizer('regex', True)
                vectorizer = TfidfVectorizer(
                    min_df=1, max_df=len(doc) * .95,
                    tokenizer=tokzr,
                    stop_words=stopwords.words('english'))
                vectors = vectorizer.fit_transform(docs).toarray()
                doc_texts = {i: v for i, v in enumerate(docs)}
                doc_dict = {i: v for i, v in enumerate(vectors)}
                feature_names = vectorizer.get_feature_names()
            #         idf_vals = vectorizer.idf_

                summ_scores = []  # includes tuples (mmr_score, sentence_id)
                # iterate through sentences to
                for i, s in doc_texts.iteritems():
                                                    # select them for summary
                    if len(summ_scores) > 0:
                        summ_v = ' '.join([doc_texts[e[1]]
                                           for e in summ_scores])
                        # summarization vector
                    else:
                        summ_v = ''
                    if summ_v != '':
                        summ_v = vectorizer.transform(
                            [summ_v]).toarray()[0]  # to tf-idf
                    score = -1 * self._mmr(
                        vectorizer.transform(
                            [s]).toarray()[0], doc_dict[len(doc_dict) - 1],
                        summ_v, self.lmbda, self.cossim)
                    if len(summ_scores) < max_length / 30 + 3:
                        # max heap data structure for mmr
                        heappush(summ_scores, (score, i))
                    else:  # Get rid of lowest score
                        heappushpop(summ_scores, (score, i))
                print summ_scores
                final_sum = []
                for s in summ_scores:
                    if self.w_t.count_words(final_sum) < max_length:
                        final_sum.append(doc_texts[s[1]])
            #         print 'before: %d' % self.w_t.count_words(final_sum)
                if self.w_t.count_words(final_sum) > max_length:
                    tmp = final_sum.pop()
                if self.w_t.count_words(final_sum) == 0:
                    final_sum.append(tmp)
            #         print 'after: %d' % self.w_t.count_words(final_sum)
                return final_sum

            def _mmr(self, s, D, Summ, lmbda, sim):
                '''
                s: Sentence for evaluation
                D: The whole document
                Summ: The summary
                lmbda: Lambda parameter
                sim: The similarity function

                Returns
                ------
                float
                '''
                if Summ == '':
                    return lmbda * sim(s, D)
                return lmbda * sim(s, D) - (1 - lmbda) * (sim(s, Summ))

            vals = []
            idx = 0
            l = sum([len(e) for e in cluster.values()])
            while (len(vals) < len(cluster.keys()) * 3) and idx < l:
                for e in cluster.values():
                    if idx < len(e):
                        vals.append(e[idx])
                idx += 1
            final_sum = summarize1(
                vals, max_length)
            
        elif mode=='knapsack':
            

        return final_sum

    def cossim(self, a, b):
        return (np.inner(a, b) / (LA.norm(a) * LA.norm(b)))

    def modified_cosine(self, sentence1, sentence2, tf1, tf2, idf_metrics):
        common_words = frozenset(sentence1) & frozenset(sentence2)

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term] * tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t] * idf_metrics[t])**2 for t in sentence1)
        denominator2 = sum((tf2[t] * idf_metrics[t])**2 for t in sentence2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    def get_tokenizer(self, alg='regex', stem=False):
        '''
        Tokenizes a string

        Args:
            text(str)
            alg(str) -- tokenization algorithm, valid options:
                regex, word
            stem(bool)
        '''
        w_t = WordTokenizer(alg=alg, stem=stem)
        return w_t

    def tokenize1(self, text, alg='regex', stem=False):
        '''
        Tokenizes a string

        Args:
            text(str)
            alg(str) -- tokenization algorithm, valid options:
                regex, word
            stem(bool)
        '''
        w_t = WordTokenizer(alg=alg, stem=stem)
        return w_t(text)

    def dump_data(self, data, **kwargs):
        if not hasattr(self, 'dumped'):
            self.dumped = {}

        for k, data in kwargs.iteritems():
            fn = '%s_%s_%s.pickle' % (k, hash_obj(data), self.dumped())
            cache_path = os.path.join(self.cachedir, fn)
            if not os.path.exists(cache_path):
                with file(cache_path, 'wb') as f:
                    try:
                        pickle.dump(data, f)
                    except Exception, e:
                        print e
                self.dumped[k] = cache_path

    def load_data(self):
        return {k: pickle.load(file(self.dumped[k]))
                for k in self.dumped}

    def _parse_args(self):
        """ Parse the Summarizer-specific parameters (as specified by
            method_opts) and add them to self.opts. Note: if an
            option is already specified in self.opts, it is NOT
            overwritten by this method.

            Notes
            ----------
            This method is NOT meant to be overwritten.
        """
        # replaces every char in every element of method_opts
        # that is not a dash, letter or digit with a dash.
        # It also convert the option to lowercase.
        fix_opts_names = re.compile(r'[^A-Za-z0-9\-]').sub
        self.method_opts = {fix_opts_names('-', k).lower(): v
                            for k, v in self.method_opts.iteritems()}

        method_parser = ArgumentParser()

        # Giant hack to overcome the fact that you can't unpack
        # with the star operator a vocabulary that has 'type'
        # as key. However, since ArgumentParser.add_argument
        # requires a 'type' parameter to specify the type of
        # an argument, I use a separate dictionary to collect
        # all the types that are specified in method_opts
        # and add them later by poking inside method_parser.
        types_map = {}

        for opt, values in self.method_opts.iteritems():
            types_map[opt] = values.pop('type', None)
            method_parser.add_argument(('--%s' % opt), **values)

        # Poking inside method_parser for the reasons described
        # above. The try...except sequence is necessary for when
        # opts is the help option, which is not present in types_map.
        for opt in (vars(o) for o in vars(method_parser)['_actions']):
            try:
                dest = opt['dest'].replace('_', '-')
                opt['type'] = types_map[dest]
            except KeyError:
                pass

        mopts, margs = method_parser.parse_known_args(self.args)
        self.args = margs
        for opt, value in vars(mopts).iteritems():
            vars(self.opts).setdefault(opt, value)


class KnapSack():

    def total_value(self, items, max_weight):
        return sum([x[2] for x in items]) if sum([x[1] for x in items]) < max_weight else 0

    cache = {}

    def solve(self, items, max_weight):
        if not items:
            return ()
        if (items, max_weight) not in self.cache:
            head = items[0]
            tail = items[1:]
            include = (head,) + self.solve(tail, max_weight - head[1])
            dont_include = self.solve(tail, max_weight)
            if self.total_value(include, max_weight) > self.total_value(dont_include, max_weight):
                answer = include
            else:
                answer = dont_include
            self.cache[(items, max_weight)] = answer
        return self.cache[(items, max_weight)]

    items = (
        ("map", 9, 150), ("compass", 13, 35), ("water",
                                               153, 200), ("sandwich", 50, 160),
        ("glucose", 15, 60), ("tin", 68,
                              45), ("banana", 27, 60), ("apple", 39, 40),
        ("cheese", 23, 30), ("beer", 52,
                             10), ("suntan cream", 11, 70), ("camera", 32, 30),
        ("t-shirt", 24, 15), ("trousers", 48, 10), ("umbrella", 73, 40),
        ("waterproof trousers", 42, 70), ("waterproof overclothes", 43, 75),
        ("note-case", 22, 80), ("sunglasses", 7, 20), ("towel", 18, 12),
        ("socks", 4, 50), ("book", 30, 10),
    )
    max_weight = 400
