from base import Summarizer
from collections import defaultdict
from plaintext import PlaintextParser
from tokenizers import Tokenizer
from stemmers import Stemmer
from lex_rank import LexRankSummarizer
import itertools
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
        self.s_t = SentTokenizer(offsets=False)
        self.w_t = WordTokenizer(stem=False)

    def summarize(self, extracted_refs, facet_results, max_length=250):
        '''
        Summarizes the extracted references based on the facet results

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
                [e.replace('\\', '') for e in t[0]['sentence']])

        summarizer = LexRankSummarizer(Stemmer('english'))

        final_summ = defaultdict(lambda: defaultdict(dict))
        ret_summ = defaultdict(list)
        counts = defaultdict(lambda: defaultdict(dict))
        for t in summaries:
            for facet in summaries[t]:
                if len(summaries[t][facet]) > 1:
                    summs = list(
                        itertools.chain.from_iterable(summaries[t][facet]))
                    parser = PlaintextParser.from_string(
                        ' '.join(summs), Tokenizer('english'))
                    summ = summarizer(parser.document, max_length)
                    final_summ[t][facet] = [unicode(sent) for sent in summ]
                    counts[t][facet] = len(final_summ[t][facet])
                else:
                    final_summ[t][facet] = self.s_t(summaries[t][facet][0])
            i = 0
            ret_summ[t] = self.pick_from_cluster(
                final_summ[t], max_length=max_length, lmb=0.3)

#         summ = defaultdict(list)
#         tokzer = WordTokenizer(stem=False)
#         for k in final_summ:
#             i = 0
#             while tokzer.count_words(summ[k]) < max_length:
#                 for f in final_summ[k]:
#                     if len(final_summ[k][f]) > i and\
#                             tokzer.count_words(summ[k]) < max_length:
#                         summ[k].append(final_summ[k][f][i])
        return ret_summ
