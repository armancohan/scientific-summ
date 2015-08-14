#!/usr/bin/python
# -*- coding: utf-8 -*-

from method.method_interface import MethodInterface
from util.es_interface import ESInterface as ESInterface
from util.es_interface_auth import ESInterface as ESAuth
import re
import os
import timeit
import json
import codecs
import numpy as np
from math import log
STOPWORDS_PATH = 'data/stopwords.txt'

# wikipedia_20131104


class Method(MethodInterface):

    """ Produce reference text by submitting the
        citance to the ElasticSearch server.
    """
    method_opts = {'maxsize': {'type': int, 'default': 100},
                   'thresh': {'type': int, 'default': False},
                   'stopwords-path': {'default': STOPWORDS_PATH},
                   'remove-stopwords': {'default': True,
                                        'action': 'store_true'},
                   'combine': {'default': False, 'action': 'store_true'},
                   'cache-path': {'default': 'cache'},
                   'idf_index': {'default': 'pubmed'}}

    def __init__(self, args, opts):
        super(Method, self).__init__(args, opts)
        self.es_int = ESInterface(host=self.opts.server,
                                  port=self.opts.port,
                                  index_name=self.opts.index_name)
        self.regex_citation = re.compile(r"(\(\s?(([A-Za-z]+\.?\s?)+,? \d+"
                                         r"(\s?([;,]|and)\s)?)+\))|"
                                         r"(\[(\d+([,–]\s?)?)+\])|"
                                         r"\[[\d,-]+\]").sub
        self.all_digits = re.compile(r"^\d+$").search
        if self.opts.remove_stopwords:
            with file(self.opts.stopwords_path) as f:
                self.stopwords = frozenset([l.strip().lower() for l in f])
        else:
            self.stopwords = frozenset([])

    def run(self, test_data):
        out_results = []
        doc_freq_path = os.path.join(self.opts.cache_path, 'idfidx' +
                                     self.opts.idf_index +
                                     'wp_doc_freq.json')
        if os.path.exists(doc_freq_path):
            with codecs.open(doc_freq_path,
                             'rb',
                             'UTF-8') as mf:
                doc_freq = json.load(mf)
        else:
            doc_freq = {}
        es_int2 = ESAuth(host='devram4.cs.georgetown.edu',
                              index_name=self.opts.idf_index)
        count_docs = es_int2.count(query='*:*')
        for ann in test_data:
            doc_type = '_'.join((ann['topic_id'].lower(),
                                 ann['reference_article'][:-4].lower()))
            doc_type = doc_type.replace(',', '').replace("'", '"')

            authors = set((ann['reference_article'][:-4].lower().strip(),
                           ann['citing_article'][:-4].lower().strip()))

            # preprocess (removes citations) and tokenizes
            # citation text before submitting to elasticsearch
            q = self.regex_citation('', ann['citation_text'])
            q = q.encode('ascii', 'ignore')
            terms = []
            for t in self.es_int.tokenize(q, 'sentence'):
                if (t not in self.stopwords and
                        t not in authors and
                        not(self.all_digits(t))):
                    if t not in doc_freq.keys():
                        count = es_int2.count(t)
                        if count > 0:
                            idf = log(count_docs / float(count + 1))
                            doc_freq[t] = idf
                            terms.append(t)
                    else:
                        idf = doc_freq[t]
                        terms.append(t)
            avg_idf = np.average([doc_freq[t] for t in terms])
            thresh = avg_idf if self.opts.thresh is not None\
                else self.opts.thresh
            q = ' '.join([t for t in terms
                          if (doc_freq[t] > thresh)])
            if q == '':
                max_idf = -1
                for t in terms:
                    if max_idf < doc_freq[t]:
                        max_idf = doc_freq[t]
                        q = t
            r = self.es_int.simple_search(q, maxsize=self.opts.maxsize,
                                          source_fields=['offset', 'sentence'],
                                          field='sentence',
                                          doc_type=doc_type)
            for e in r:
                fld = e.pop('fields')
                e['offset'] = [eval(fld['offset'][0])]
#                 beg = e['offset'][0][0] - \
#                     100 if e['offset'][0][0] else e['offset'][0][0]
#                 end = e['offset'][0][1] + 100
#                 e['offset'] = [(beg, end)]
                e['sentence'] = fld['sentence'][0]
                e['query'] = q

            if self.opts.combine:
                if len(r) == 0:
                    r = [{'_type': doc_type,
                          '_index': self.opts.index_name,
                          '_score': 0,
                          'sentence': '',
                          'offset': [(0, 1)],
                          'query':q, '_id':-11}]
                r = [{'_type': r[0]['_type'],
                      '_index': r[0]['_index'],
                      'query': q,
                      'topic': ann['topic_id'].lower(),
                      'citance_number': ann['citance_number'],
                      'citation_text': ann['citation_text'],
                      'citing_article': ann['citing_article'],
                      '_score': sum([e['_score'] for e in r]),
                      'offset': [e['offset'][0] for e in r],
                      'sentence': [e['sentence'] for e in r],
                      '_id': '-000001'}]
            out_results.append(r)
        with codecs.open(doc_freq_path,
                         'wb',
                         'UTF-8') as mf:
            json.dump(doc_freq, mf, indent=2)
        return out_results
