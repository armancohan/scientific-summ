'''
Created on May 9, 2015

@author: rmn
'''
from base import Summarizer
from collections import defaultdict
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from lex_rank import LexRankSummarizer
import itertools
from log_conf import Logger
from util.tokenization import WordTokenizer, SentTokenizer
from random import randint

logger = Logger('.'.join(__file__.split('/')[-2:-1])).logger


class Summarizer(Summarizer):
    '''
    classdocs
    '''

    def __init__(self, args, opts):
        '''
        Constructor
        '''
        self.s_t = SentTokenizer(offsets=False)
        self.w_t = WordTokenizer(stem=False)

    def summarize(self, extracted_refs, facet_results, max_length=250):
        '''
        Summarizes the extracted references based on the facet results

        Uses LexRank to choose the most salient sentences from
            reference sentences in each facet

        Args:
            extracted_refs(list) -- results of the method.run (e.g. simple.py)
            facet_results(dict) -- facets for each extracted reference
                Look at data/task1b_results1.json
            max_length(int) -- maximum length of the summary
        '''
        summaries = defaultdict(list)
        for t in extracted_refs:
            topic = t[0]['topic']
            if isinstance(t[0]['sentence'][0], list):
                logger.warn('Unexpected, should check')
            summaries[topic.upper()].append(t[0]['citation_text'])

        for t in summaries:
            while self.w_t.count_words(summaries[t]) > max_length:
                i = randint(0, len(summaries[t])-1)
                summaries[t].pop(i)
        return summaries
