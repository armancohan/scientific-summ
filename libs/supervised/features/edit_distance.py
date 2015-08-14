'''
Created on Feb 10, 2015

@author: rmn
'''
import distance
from libs.supervised.feature_interface import FeatureInterface


class Feature(FeatureInterface):

    '''
    Extracts edit distance feature between two sentences
    '''

    def __init__(self):
        '''
        Constructor
        '''
        pass

    def extract(self, query, document, params=None):
        '''
        Extracts features for a given pair of query and document

        Args:
            query(str): The query string
            document(str)
            params: parameters for the feature extractor

        Returns
            float -- feature value
        '''
        return distance.levenshtein(self.tokenize(query),
                                    self.tokenize(document),
                                    normalized=True)


if __name__ == '__main__':
    f = Feature()
    print f.extract('Existing methods for ranking aggregation includes unsupervised learning methods such as BordaCount and Markov Chain, and supervised learning methods such as Cranking.',
                    'Markov Chain based ranking aggregation assumes that there exists a Markov Chain on the' +
                    'objects. The basic rankings of objects are utilized to construct the Markov Chain,')