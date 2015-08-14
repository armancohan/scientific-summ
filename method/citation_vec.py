#!/usr/bin/python
# -*- coding: utf-8 -*-

from method.method_interface import MethodInterface
from util.es_interface import ESInterface
from nltk.tokenize.regexp import RegexpTokenizer
import re
from argparse import ArgumentParser
import codecs
import json
import sys

STOPWORDS_PATH = 'data/stopwords.txt'


class Method(MethodInterface):

    """ Produce reference text by submitting the
        citance to the ElasticSearch server.
    """
    method_opts = {'maxsize': {'type': int, 'default': 3},
                   'stopwords-path': {'default': STOPWORDS_PATH},
                   'remove-stopwords': {'default': False,
                                        'action': 'store_true'},
                   'combine': {'default': False, 'action': 'store_true'},
                   'analyzer': {'default': False, 'type': str},
                   'ngram': {'default': False, 'type': int}}

    def __init__(self, args, opts):
        super(Method, self).__init__(args, opts)
        self.es_int = ESInterface(host=self.opts.server,
                                  port=self.opts.port,
                                  index_name=self.opts.index_name)

        self.regex_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
                                         r"(\[(\d+([,–]\s?)?)+\])|"
                                         r"\[[\d,-]+\]").sub
        self.all_digits = re.compile(r"^\d+$").search
        if self.opts.stopwords_path:
            stop_path = self.opts.stopwords_path
        else:
            stop_path = STOPWORDS_PATH
        if self.opts.remove_stopwords:
            with file(self.opts.stopwords_path) as f:
                self.stopwords = frozenset([l.strip().lower() for l in f])
        else:
            self.stopwords = frozenset([])
        self.tokenizer = RegexpTokenizer('[^\w\-\']+', gaps=True)

    def run(self, test_data):
        #         with codecs.open('tmp/test_data.json', 'wb', 'utf-8') as mf:
        #             json.dump(test_data, mf, indent=2)
        out_results = []
        det_res = {}
        for ann in test_data:
            doc_type = '_'.join((ann['topic_id'].lower(),
                                 ann['reference_article'][:-4].lower()))
            # TEMPORARY FIX FOR WRONG DOCUMENT TYPE NAME
            doc_type = doc_type.replace('train', 'eval')
            doc_type = doc_type.replace(',', '').replace("'", '"')

            # TEMPORARY FIX FOR WRONG DOCUMENT TYPE NAME
            doc_type = doc_type.replace('eval', 'train')

            authors = set((ann['reference_article'][:-4].lower().strip(),
                           ann['citing_article'][:-4].lower().strip()))

            # preprocess (removes citations) and tokenizes
            # citation text before submitting to elasticsearch
            q = self.regex_citation('', ann['citation_text'])
            q = q.encode('ascii', 'ignore')
#             tokens = self.es_int.tokenize(q, "sentence")
            tokens = self.tokenizer.tokenize(q)
            tokens = ['"' + t + '"' if '-' in t else t for t in tokens]
            q = ' '.join([t for t in tokens
                          if (t not in self.stopwords and
                              t not in authors and
                              not(self.all_digits(t)))])

            if self.opts.ngram:
                tokens = self.es_int.tokenize(q, "sentence")
                new_query = ''
                for i in range(len(tokens) - self.opts.ngram):
                    tmp = ''
                    for j in range(i, i + self.opts.ngram):
                        tmp += tokens[j] + ' '
                    new_query += '"' + tmp.strip() + '" '
                q = new_query.strip()
#             q = '*:*'
            if self.opts.analyzer:
                r = self.es_int.simple_search(q, maxsize=self.opts.maxsize,
                                              source_fields=[
                                                  'offset', 'sentence'],
                                              # field='sentence',
                                              doc_type=doc_type,
                                              params={'analyzer': self.opts.analyzer})
            else:
                r = self.es_int.simple_search(q, maxsize=self.opts.maxsize,
                                              source_fields=[
                                                  'offset', 'sentence'],
                                              # field='sentence',
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
                e['topic'] = ann['topic_id'].lower()

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
#         with codecs.open('tmp/out_results.json', 'wb', 'utf-8') as mf:
#             json.dump(out_results, mf, indent=2)
#         sys.exit()
        return out_results


def parse_args(args):
    ap = ArgumentParser()
    ap.add_argument('--remove-stopwords', dest='remove_stopwords',
                    action='store_true'
                    )
    ap.add_argument('--out-file', dest='out_file',
                    default=False,
                    help='generate detailed output of the result')
    ap.add_argument('--out-dir', dest='out_dir',
                    default=False,
                    help='generate several files for each result'
                    ' in the out_dir')
    return ap.parse_known_args(args)
