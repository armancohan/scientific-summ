'''
Summarizing by detecting communities in graph of references.

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
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import community
import cPickle

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
        Summarizes the extracted references based on community detection

        Uses LexRank to choose the most salient sentences from
            reference sentences in each facet

        Args:
            extracted_refs(list) -- results of the method.run (e.g. simple.py)
            facet_results(dict) -- facets for each extracted reference
                Look at data/task1b_results1.json
            max_length(int) -- maximum length of the summary

        '''
        citances = defaultdict(list)
        summarizer = LexRankSummarizer(Stemmer('english'))
        summary = defaultdict(lambda: defaultdict(list))
        for t in extracted_refs:
            citances[t[0]['topic']].append(
                {'refs': t[0]['sentence'],
                 'citance': self.clean_citation(t[0]['citation_text'])})

        for topic, citance in citances.iteritems():
            # Create graph of citation similarities
            vectorizer = TfidfVectorizer(
                tokenizer=self.tokenize, min_df=1, max_df=len(citances) * .9)
            sent_tokenize = SentTokenizer(offsets=False)
            cit_vectors = vectorizer.fit_transform(list(itertools.chain.from_iterable(
                [sent_tokenize(e['refs']) for e in citance]))).toarray()
            cit_dict = {i: v for i, v in enumerate(cit_vectors)}
            cit_text = {i: v for i, v in enumerate(list(itertools.chain.from_iterable(
                [sent_tokenize(e['refs']) for e in citance])))}
            cits = []
            for e in cit_dict:  # vector (numpy array)
                for e1 in cit_dict:
                    if e != e1:
                        simil = self.cossim(cit_dict[e],
                                            cit_dict[e1])
                        if simil > 0.1:
                            cits.append((e, e1, simil))
            G = nx.Graph()
            G.add_weighted_edges_from(cits)
            with open('tmp/graph-%s.cpickle' % topic, 'wb') as mf:
                cPickle.dump(G, mf)
            part = community.best_partition(G)
            clusters = defaultdict(list)

            for k, v in part.iteritems():
                clusters[v].append(cit_text[k])
            # clusters includes ref sentences that belong in each cluster
            # Find the most salient sentence in each cluster
            sal_in_cluster = {}  # salient sentences for each cluster
            for i in clusters:
                parser = PlaintextParser.from_string(
                    ' '.join(clusters[i]).replace('\\', ''), Tokenizer('english'))
                summ = summarizer(parser.document, 5)
                # 5 is the number of sentences returned by LexRank
                sal_in_cluster[i] = [unicode(s) for s in summ]
                # The most salient sentences in each cluster
            summary[topic.upper()] =\
                self.pick_from_cluster(
                    sal_in_cluster, max_length, weighted=False)
        return summary

if __name__ == "__main__":
    import codecs
    import json
    with codecs.open('/home/rmn/dev/git/naacl/data/task2_citations.json', 'rb', 'utf-8') as mf:
        data = json.load(mf)
    import itertools
    from pprint import pprint
    citations = []
    c = data['D1418_TRAIN'].values()
    citations = list(itertools.chain.from_iterable(c))
    cl = Summarizer()
    pprint(cl.summarize(citations, 10))
