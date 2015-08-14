'''
Created on Sep 19, 2014

@author: rmn
'''
from rerank.null import Reranker as RerankInterface
import json
import codecs
import os
import sys
from copy import deepcopy
from libs.evaluate import merge_offsets
from libs.supervised.prep.prepare import Prep
from constants import get_path, join_path
from libs.supervised.classifiers.svm_rank import Supervised
from util.common import hash_obj
from util.cache import simple_caching, object_hashing
from importlib import import_module
import constants

path = get_path()
STOPWORDS_PATH = path['data'] + '/stopwords.txt'
CLF_PATH = join_path(
    path['root'], 'libs/supervised/classifiers')
docs_path = join_path(path['data'], 'TAC_2014_BiomedSumm_Training_Data')
json_data_path = join_path(path['data'], 'v1-2a.json')

# root_proj_path = os.getcwd()
# while not('.git' in os.listdir(root_proj_path)):
#     root_proj_path = os.path.split(root_proj_path)[0]
# if not(root_proj_path in sys.path):
#     sys.path.append(root_proj_path)


class Reranker(RerankInterface):

    reranker_opts = {'cutoff': {'type': int, 'default': 4},
                     'lookup': {'type': int, 'default': 4},
                     'relaxation': {'type': int, 'default': 0},
                     'probability': {'action': 'store_true'},
                     'features': {'nargs': '+', 'type': str, 'help': 'Space separated features '
                                  'Available options: '
                                  'lcs, bm25, edit_distance, topic_model', 'default': ['lcs', 'bm25', 'edit_distance', 'topic_model']}}

    def __init__(self, args=None, opts=None):
        """ Initialize reranker.
        """
        super(Reranker, self).__init__(args, opts)
#         self.es_int = ESInterface(host=self.opts.server,
#                                   port=self.opts.port,
#                                   index_name=self.opts.index)
#         self.qsdata = json.load(file(self.opts.mlefn))
        self.models = {}  # svm models for each topic
        self.all_docs = {}  # keeps the set of all documents for each topic
        self.extractors = []
        p = Prep(index='biosum')
        self.train_raw = p.prep(docs_path, json_data_path)
        for f in self.opts.features:
            base_module = import_module('libs.supervised.features.%s' % f)
            if f != 'topic_model':
                base_clf = base_module.Feature()
                self.extractors.append(base_clf)

    def train(self, train_set):
        '''
        Train_set is null,
        the training data come from docs_path and json_data_path

        '''
        @object_hashing(
            cache_comment='svm_models_%s' % hash_obj(
                train_set),
            cachedir=constants.get_path()['cache'])
        def _train(train_raw):
            models = {}
            all_docs = {}
            for topic in train_raw:
                x_train = []
                y_train = []
                for inst in train_raw[topic]:

                    feature_vector = [
                        ext.extract(inst[0], inst[1]) for ext in self.extractors]
                    x_train.append(feature_vector)
                    y_train.append(inst[2])
                svm = Supervised(self.args, self.opts)
                with open(constants.get_path()['tmp'] + '/ltr-features-%s' % topic, 'wb') as mf:
                    json.dump(
                        {'x_train': x_train, 'y_train': y_train}, mf, indent=2)
                svm.train(x_train, y_train)
                models[topic.lower()] = svm
                all_docs[topic] = [inst[1] for inst in train_raw[topic]]
                return models, all_docs
        self.models, self.all_docs =\
            _train(train_set)

    def rerank(self, results):
        '''
        format of the results:
        [ [{_type, _index, sentence,offset]
          [{...}],
          ...
          []
        ]
        '''
#         with codecs.open('tmp/results.json', 'wb', 'utf-8') as mf:
#             json.dump(results, mf, indent=2)
        out = deepcopy(results)

        if 'topic_model' in self.opts.features:
            print self.train_raw.keys()
            base_module = import_module('libs.supervised.features.topic_model')
            f = base_module.Feature(
                [inst[1] for inst in self.train_raw[results[0][0]['topic'].upper()]])
            self.extractors.append(f)

        self.train(train_set=self.train_raw)

        topop = []
        # TOPIC_MODEL (LSA) FEATURE
        # Needs different initialization, that's why it is defined here

        for res in results:
            for i in range(len(res[0]['offset'])):
                for s in res[0]['sentence']:
                    feature_vector = [
                        ext.extract(res[0]['query'], s) for ext in self.extractors]
                    svm = self.models[res[0]['topic']]
                    print svm.run(feature_vector)
                    import pdb
                    pdb.set_trace()
                print '----'
#                     svm.predict(feature_vector)

                for j in range(i + 1, min(self.opts.lookup +
                                          self.opts.cutoff,
                                          len(res[0]['offset']))):
                    # look at lower ranked results, if there are overlaps,
                    # merge
                    newspan = merge_offsets(res[0]['offset'][i],
                                            res[0]['offset'][j],
                                            relaxation=self.opts.relaxation)
                    if newspan is not None:
                        res[0]['offset'][i] = newspan
                        topop.append(j)
            res[0]['offset'] = [
                x for idx1, x in enumerate(res[0]['offset']) if idx1 not in topop]
            res[0]['offset'] = res[0]['offset'][0:self.opts.cutoff]
        return results
#     def _process_documents(self, results):
#         pass
if __name__ == '__main__':
    import codecs
    import json
    with codecs.open('/home/rmn/Downloads/tmp.json', 'rb', 'utf-8') as mf:
        results = json.load(mf)
    rrnk = Reranker()
    newres = rrnk.rerank(results)
    with codecs.open('/home/rmn/Downloads/tmp1.json', 'wb', 'utf-8') as mf:
        json.dump(newres, mf, indent=2)
