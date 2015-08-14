'''
Created on Feb 18, 2015

@author: rmn

Wordnet (Thesaurus) based similarity measures
'''
from __future__ import division
from libs.supervised.feature_interface import FeatureInterface
from nltk.corpus import wordnet as wn


class Feature(FeatureInterface):

    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def extract(self, query, document, no_stopwords=True):
        '''
        Extracts features for a given pair of query and document

        Args:
            query(str): The query string
            document(str)
            params: parameters for the feature extractor

        Returns
            float -- feature value
        '''
        return len(self._lcs(self.tokenize(query, no_stopwords=no_stopwords),
                             self.tokenize(document, no_stopwords=no_stopwords)))

    def _get_simil_term(self, x, y, mode='lch'):
        '''
        Returns the similarity between two terms x and y
        Args:
            x, y (str)
            mode = lch | path | wup
        '''
        w1 = wn.synsets(x)
        w2 = wn.synsets(y)
        if len(w1) == 0 or len(w2) == 0:
            return 0
        else:
            if mode == 'lch':
                return max([wn.lch_similarity(e1, e2) for e1 in w1 for e2 in w2 if e1.pos == e2])
            elif mode == 'path':
                return max([wn.path_similarity(e1, e2) for e1 in w1 for e2 in w2 if e1.pos == e2])
            elif mode == 'wup':
                return max([wn.wup_similarity(e1, e2) for e1 in w1 for e2 in w2 if e1.pos == e2])

    def _lcs(self, x, y):
        n = len(x)
        m = len(y)
        table = dict()  # a hashtable, but we'll use it as a 2D array here

        for i in range(n + 1):     # i=0,1,...,n
            for j in range(m + 1):  # j=0,1,...,m
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif x[i - 1] == y[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])

        # Now, table[n, m] is the length of LCS of x and y.

        # Let's go one step further and reconstruct
        # the actual sequence from DP table:

        def recon(i, j):
            if i == 0 or j == 0:
                return []
            elif x[i - 1] == y[j - 1]:
                return recon(i - 1, j - 1) + [x[i - 1]]
            # index out of bounds bug here: what if the first elements in the
            # sequences aren't equal
            elif table[i - 1, j] > table[i, j - 1]:
                return recon(i - 1, j)
            else:
                return recon(i, j - 1)

        return recon(n, m) / min(m, n)

if __name__ == "__main__":
    lcs = Feature()
    # test 1
    ret = lcs.extract(
        'This is a rediculous idea', 'is not what idea')
    print ret
'''
Created on Apr 5, 2015

@author: rmn
'''
