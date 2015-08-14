'''
Created on Feb 10, 2015

@author: rmn
'''
from abc import abstractmethod, ABCMeta
from util.common import tokenize


class FeatureInterface(object):

    '''
    Feature base class, includes methods for extracting
        features from a set of query and document
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        '''
        Constructor
        '''

    def tokenize(self, doc, stem=True, no_stopwords=True):
        '''
        Tokenizes a document into a list of terms

        Args:
            doc(str)
            stem(bool) -- Do stemming?
            no_stopwords(bool) -- No stopwords in output?

        Returns:
            str
        '''
        return tokenize(doc, no_stopwords=no_stopwords, stem=stem)

    @abstractmethod
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
        pass
