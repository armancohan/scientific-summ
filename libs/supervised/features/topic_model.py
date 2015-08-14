'''
Created on Feb 17, 2015

@author: rmn
'''
from gensim import models
from util.common import hash_obj
import os
from libs.supervised.feature_interface import FeatureInterface
'''
Created on Feb 10, 2015

@author: rmn
'''
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.lsimodel import LsiModel
from sklearn.metrics.pairwise import cosine_similarity
from constants import get_path

path = get_path()
CACHE_DIR = path['cache']


class Feature(FeatureInterface):

    '''
    Extracts edit distance feature between two sentences
    '''

    def _build_model(self, all_documents, remove_once=False):
        '''
        Builds the lsa model

        Returns:
            dictionary, corpus
        '''
        doc_hash = hash_obj(all_documents)
        corp_cache_path = CACHE_DIR + '\\' +\
            '_corp_' + doc_hash + str(int(remove_once))
        dic_cache_path = CACHE_DIR + '\\' +\
            '_dic_' + doc_hash + str(int(remove_once))
        lsi_cache_path = CACHE_DIR + '\\' +\
            '_lsi_' + doc_hash + str(int(remove_once))
        if os.path.exists(corp_cache_path) \
                and os.path.exists(dic_cache_path)\
                and os.path.exists(lsi_cache_path):
            lsi = models.LsiModel.load(lsi_cache_path)
            corp = MmCorpus(corp_cache_path)
            dic = Dictionary.load(dic_cache_path)
        else:
            texts = [self.tokenize(doc) for doc in all_documents]
            all_tokens = sum(texts, [])
            if remove_once:
                tokens_once = set(word for word in set(all_tokens)
                                  if all_tokens.count(word) == 1)
                texts = [[word for word in text if word not in tokens_once]
                         for text in texts]
            dic = Dictionary(texts)
            corp = [dic.doc2bow(text) for text in texts]

            MmCorpus.serialize(corp_cache_path, corp)
            dic.save(dic_cache_path)
            lsi = models.LsiModel(
                corp, id2word=dic, num_topics=20)
            lsi.save(lsi_cache_path)
        return dic, corp, lsi

    def __init__(self, all_documents, num_topics=20, remove_once=False):
        '''
        Initializes the LSA model

        Args:
            all_documents(list(str)): a list of documents
            num_topics(int): number of LSA topics
            remove_once(bool): Remove the words that occur only once
        '''
        # word mapping for the returned results
        self._dictionary, self._corpus, self._lsi = self._build_model(
            all_documents, remove_once)

    def extract(self, query, document):
        '''
        Extracts LSA similarity features for a given
         pair of query and document
         i.e. The cosine similarity between the LSA vectors
             of the query and the document

        Args:
            query(str): The query string
            document(str)

        Returns
            float -- feature value
        '''
        query_bow = self._dictionary.doc2bow(
            self.tokenize(query))
        doc_bow = self._dictionary.doc2bow(
            self.tokenize(document))
        query_lsi = self._lsi[query_bow]
        doc_lsi = self._lsi[doc_bow]
        if len(query_lsi) == 0 or len(doc_lsi) == 0:
            return 1
        else:
            return cosine_similarity([y for (_, y) in query_lsi],
                                     [y for (_, y) in doc_lsi])[0][0]


if __name__ == '__main__':
    f = Feature(['In machine learning, the problem of unsupervised learning is that of trying to find hidden structure in unlabeled data.', 'Since the examples given to the learner are unlabeled, there is no error or reward signal to evaluate a potential solution.', 'This distinguishes unsupervised learning from supervised learning and reinforcement learning.',
                 'ranking aggregation is closely related to the problem of density estimation in statistics.', '[1] However unsupervised learning also encompasses many other techniques that seek to summarize and explain key features of the data.', 'Many methods employed in unsupervised learning are based on data mining methods used to preprocess[citation needed] data.'])
    print f.extract('Existing methods for ranking aggregation includes unsupervised learning methods such as BordaCount and Markov Chain, and supervised learning methods such as Cranking.',
                    'Markov Chain based ranking aggregation assumes that there exists a Markov Chain on the' +
                    'objects. The basic rankings of density are utilized to construct the Markov Chain,')
