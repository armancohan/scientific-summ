'''
Created on Feb 4, 2015

@author: rmn
'''
from __future__ import division
from util.es_interface import ESInterface
from libs.supervised.feature_interface import FeatureInterface
count_total = -1
avg_doc_length = -1
import constants


class Feature(FeatureInterface):

    def __init__(self, index_name=constants.bm25_index):
        '''
        default constuctor

        Args:
            index_name(str): The elasticsearch index name that will
                be used to retrieve documents and idfs
        '''
        self.es_int = ESInterface(index_name=index_name)
        print self.es_int.get_avg_size('sentence')
        self.avg_doc_length = -1

    def extract(self, query, document, stem=True, no_stopwords=True, b=0.75, k1=1.25):
        '''
        Args:
            query(str)
            document(str)
            stem(bool)
            no_stopwords(bool)
            b(float): Controls to what degree document length normalizes tf values.
            k1(float): Controls non-linear term frequency normalization
        '''
        q_terms = list(set([w for w in self.tokenize(
            query, stem=stem, no_stopwords=no_stopwords)]))
        d_terms = list(set([w for w in self.tokenize(
            document, stem=stem, no_stopwords=no_stopwords)]))
        d_len = len(self.tokenize(document, stem=False, no_stopwords=False))
        if self.avg_doc_length == -1:
            self.avg_doc_length = self.es_int.get_avg_size('sentence')
        score = 0
        for t in q_terms:
            score += self.es_int.get_idf(t) *\
                ((self._freq(t, d_terms) * (k1 + 1)) /
                 (self._freq(t, d_terms) +
                  k1 * (1 - b + b * (d_len / avg_doc_length))))
        return score

    def _freq(self, term, doc):
        '''
        Gets the frequency of a term in a doc

        Args:
            term(str)
            doc(list(str)) -- list of strings
        '''
        return len([1 for t in doc if t == term])

if __name__ == '__main__':
    f = Feature('biosum')
    print f.extract('Existing methods for ranking aggregation includes unsupervised learning methods such as BordaCount and Markov Chain, and supervised learning methods such as Cranking.',
                    'Markov Chain based ranking aggregation assumes that there exists a Markov Chain on the' +
                    'objects. The basic rankings of objects are utilized to construct the Markov Chain,')
