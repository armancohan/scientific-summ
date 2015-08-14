'''
Created on May 4, 2015

@author: rmn
'''
from __future__ import division
import codecs
import json
from summarizer.base import Summarizer
from copy import deepcopy
import random
from util.tokenization import WordTokenizer


class Summarizer(Summarizer):
    '''
    Random selection of the sentences based on citations

    '''

    def __init__(self):
        self.w_t = WordTokenizer(stem=False)

    def summarize(self, citations, max_length=250):
        '''
        Randomly select from citations

        Args:
            citations(list)
                A list of strings
            max_length(int)
                maximum length of the summary in words

        Returns
        -------
        List
            A list of ranked strings for the final summary
        '''
        final_sum = deepcopy(citations)
        while self.w_t.count_words(final_sum) > max_length:
            i = random.randint(0, len(final_sum) - 1)
            del final_sum[i]
        final_sum.pop()
        return final_sum


