'''
Created on Feb 18, 2015

@author: rmn
'''
from __future__ import division
from libs.supervised.feature_interface import FeatureInterface


class Feature(FeatureInterface):

    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def extract(self, query, document, no_stopwords=False):
        '''
        Extracts features for a given pair of query and document

        Args:
            query(str): The query string
            document(str)
            params: parameters for the feature extractor

        Returns
            float -- feature value
        '''
        q = self.tokenize(query, no_stopwords=no_stopwords)
        doc = self.tokenize(document, no_stopwords=no_stopwords)
        return len(self._lcs(q, doc)) / min(len(q), len(doc))

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

        return recon(n, m)

if __name__ == "__main__":
    lcs = Feature()
    # test 1
    ret = lcs.extract(
        'This is a rediculous idea', 'is not what idea')
    print ret
