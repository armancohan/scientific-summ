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
from util.tokenization import WordTokenizer

logger = Logger('.'.join(__file__.split('/')[-2:-1])).logger


class Summarizer(Summarizer):
    '''
    classdocs
    '''

    def __init__(self, args, opts):
        '''
        Constructor
        '''

    def summarize(self, extracted_refs, facet_results, max_length=250):
        '''
        Summarizes the extracted references based on the facet results


        Args:
            extracted_refs(list) -- results of the method.run (e.g. simple.py)
            facet_results(dict) -- facets for each extracted reference
                Look at data/task1b_results1.json
            max_length(int) -- maximum length of the summary
        '''
        summaries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for t in extracted_refs:
            topic = t[0]['topic']
            citance = t[0]['citance_number']
            if isinstance(t[0]['sentence'][0], list):
                logger.warn('Unexpected, should check')
            summaries[topic.upper()][citance]\
                [facet_results[topic.upper()]
                 [str(citance)]['SVM_LABEL']].append(
                t[0]['sentence'])
        import pdb
        pdb.set_trace()

        summarizer = LexRankSummarizer(Stemmer('english'))

        final_summ = defaultdict(lambda: defaultdict(dict))
        counts = defaultdict(lambda: defaultdict(dict))
        for t in summaries:
            for c in summaries[t]:
                for facet in summaries[t][c]:
                    summs = list(
                        itertools.chain.from_iterable(summaries[t][c][facet]))
                    parser = PlaintextParser.from_string(
                        ' '.join(summs), Tokenizer('english'))
                    summ = summarizer(parser.document, max_length)
                    final_summ[t][c][facet] = [unicode(sent) for sent in summ]
                    counts[t][c][facet] = len(final_summ[t][c][facet])
        summ = defaultdict(list)
        tokzer = WordTokenizer(stem=False)
        for k in final_summ:
            i = 0
            c = final_summ[k].keys()[0]
            while tokzer.count_words(summ[k]) < max_length:
                for c in final_summ[k]:
                    for f in final_summ[k][c]:
                        if len(final_summ[k][c][f]) > i and\
                                tokzer.count_words(summ[k]) < max_length:
                            summ[k].append(final_summ[k][c][f][i])
        return summ
