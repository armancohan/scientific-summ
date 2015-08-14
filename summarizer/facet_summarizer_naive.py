from collections import defaultdict
import itertools

from base import Summarizer
from log_conf import Logger
from util.tokenization import WordTokenizer, SentTokenizer


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

        Chooses from facets naively

        Args:
            extracted_refs(list) -- results of the method.run (e.g. simple.py)
            facet_results(dict) -- facets for each extracted reference
                Look at data/task1b_results1.json
            max_length(int) -- maximum length of the summary
        '''
        summaries = defaultdict(lambda: defaultdict(list))
        for t in extracted_refs:
            topic = t[0]['topic']
            citance = t[0]['citance_number']
            if isinstance(t[0]['sentence'][0], list):
                logger.warn('Unexpected, should check')
            summaries[topic.upper()]\
                [facet_results[topic.upper()]
                 [str(citance)]['SVM_LABEL']].append(
                t[0]['sentence'])

        final_summ = defaultdict(lambda: defaultdict(dict))
        counts = defaultdict(lambda: defaultdict(dict))
        sent_tok = SentTokenizer(offsets=False)
        for t in summaries:
            for facet in summaries[t]:
                sents = []
                for e in summaries[t][facet]:
                    sents.extend(sent_tok(e))
                final_summ[t][facet] = sents
                counts[t][facet] = len(final_summ[t][facet])
        summ = defaultdict(list)
        tokzer = WordTokenizer(stem=False)
        for k in final_summ:
            for f in final_summ[k]:
                for i in range(len(final_summ[k][f])):
                    if len(final_summ[k][f]) > i and\
                            tokzer.count_words(summ[k]) < max_length:
                        summ[k].append(final_summ[k][f][i])
        return summ
