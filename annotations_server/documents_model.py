from __future__ import print_function
import os
import codecs
import charade
from collections import OrderedDict
from itertools import chain
import sys

try:
    from util.tokenizer import para_tokenize
except ImportError:
    # BEGIN PYTHONPATH FIX #
    # appends root of project if PYTHONPATH is not set;
    # relies on the presence of .git in project root.
    root_proj_path = os.getcwd()
    root_flag = False
    while not('.git' in os.listdir(root_proj_path)) and not(root_flag):
        if root_proj_path == '/':
            root_flag = True
        root_proj_path = os.path.split(root_proj_path)[0]
        print(root_proj_path)
    if not(root_proj_path in sys.path) and not(root_flag):
        sys.path.append(root_proj_path)
    # END  PYTHONPATH  FIX #
    from util.tokenizer import para_tokenize


def listfulldir(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]


class DocumentsModel(object):

    def __init__(self, data_path, verbose=False):
        # data_path is the path to TAC_2014_BiomedSumm folder
        self.docs = {}
        self.verbose = verbose
        for topic_path in listfulldir(os.path.join(data_path, 'data')):
            topic = os.path.split(topic_path)[1].lower()
            self.docs.setdefault(topic, {})
            for doc_path in listfulldir(os.path.join(topic_path,
                                                     'Documents_Text')):
                doc = os.path.split(doc_path)[1][:-4].lower()

                with codecs.open(doc_path, mode='rb',
                                 encoding='utf-8', errors='strict') as df:
                    try:
                        self.docs[topic][doc] = df.read().replace('\r', '')
                    except UnicodeDecodeError:
                        with file(doc_path, mode='rb') as df:
                            frmt = charade.detect(df.read())['encoding']
                        with codecs.open(doc_path, mode='rb', encoding=frmt,
                                         errors='strict') as df:
                            self.docs[topic][doc] = df.read().replace('\r', '')
        if self.verbose:
            print('list of topics: %s' % '; '.join(self.docs.keys()))
            dnames = set(chain(*[d.keys() for d in self.docs.itervalues()]))
            print('list of doc_name: %s' % '; '.join(dnames))

        # create name aliases for incosistencies caused by ES
        for topic in self.docs.keys():
            for doc in self.docs[topic].keys():
                if doc.find(',') >= 0 or doc.find('\'') >= 0:
                    new_doc = doc.replace(',', '').replace('\'', '"')
                    self.docs[topic][new_doc] = self.docs[topic][doc]

        self.para_index = {}
        for topic in self.docs:
            self.para_index.setdefault(topic, {})
            for doc, data in self.docs[topic].iteritems():
                paragraphs = para_tokenize(self.docs[topic][doc])
                soff = [(s, e) for s, e in sorted(paragraphs['offsets'],
                                                  key=lambda x: x[0],
                                                  reverse=True)]
                self.para_index[topic][doc] = OrderedDict(soff)

    def get_para(self, topic_id, doc_name, interval):
        if self.verbose:
            print('request for "%s", "%s", "%s"' %
                  (topic_id, doc_name, interval))
        for s, e in self.para_index[topic_id][doc_name].iteritems():
            if interval[0] > s:
                return {'sentence': self.get_doc(topic_id, doc_name, (s, e)),
                        'offset': [s, e]}

    def get_doc(self, topic_id, doc_name, interval=None):
        if self.verbose:
            print('request for "%s", "%s", "%s"' %
                  (topic_id, doc_name, interval))
        if not interval:
            i = (None, None)
        else:
            i = interval[:]
        tid = topic_id.lower()
        did = doc_name.lower().replace('.txt', '')
        return self.docs[tid][did][i[0]:i[1]]

    def get_doc_len(self, topic_id, doc_name):
        if self.verbose:
            print('request for "%s", "%s", "%s"' %
                  (topic_id, doc_name))
        tid = topic_id.lower()
        did = doc_name.lower().replace('.txt', '')
        self.docs[tid][did]

    def get_all(self):
        '''
        Returns:
            dict:
                keys: topic_id e.g. d1418_train
                    inner_keys:    document name e.g. 'yang,rosa'
        '''
        return self.docs
