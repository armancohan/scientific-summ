import os
import sys
from math import fabs
import inspect
from hashlib import md5
import numpy as np

# BEGIN PYTHONPATH FIX #
# appends root of project if PYTHONPATH is not set;
# relies on the presence of .git in project root.
root_proj_path = os.getcwd()
while not('.git' in os.listdir(root_proj_path)):
    root_proj_path = os.path.split(root_proj_path)[0]
if not(root_proj_path in sys.path):
    sys.path.append(root_proj_path)
# END  PYTHONPATH  FIX #

from rerank.tools.rerank_tools import Timer
from rerank.clustering_lib.clustering_utils import CosineSimilarity
from util.cache import simple_caching
from util.es_interface import ESInterface
from rerank.insertion_rank_lib import expansion_methods


class Document(object):

    def __init__(self, _id, rank, tokens, relevance, dtype,
                 score, original_rank=None,
                 weights=None, sum_square_weights=None,
                 weighted=False):
        self.id = _id
        self.rank = rank
        self.tokens = tokens
        self.terms = set(tokens)
        self.relevance = relevance
        self.score = score
        self.type = dtype
        self.weighted = weighted

        if not original_rank:
            original_rank = rank
        self.original_rank = original_rank

        if not(weights):
            self.w = self._calculate_w(weights)

        if not(sum_square_weights):
            self.sum_square_w = self._calculate_ssw()

    def _calculate_ssw(self, weights=None):
        if not weights:
            self.w = self._calculate_w(weights)
        ssw = sum([(self.w[t] ** 2) for t in self.w])
        return ssw

    def _calculate_w(self, weights):
        weights = {}
        if self.weighted:
            for t in self.tokens:
                weights[t] = weights.get(t, 0) + 1
        else:
            weights = {t: 1 for t in self.tokens}
        return weights


class InsertionRankHelper(object):

    """docstring for InsertionRankHelper"""

    def __init__(self,
                 filter_list=None,
                 base_cache=None,
                 cachedir='cache',
                 eshost=None,
                 esport=None,
                 esindex=None,
                 sim_func=CosineSimilarity(),
                 stopwords=None,
                 weighted=False,
                 query_terms_only=False):

        if not eshost:
            eshost = 'localhost'
        if not esport:
            esport = 9200
        if not esindex:
            esindex = 'pubmed'

        self.es = ESInterface(host=eshost, port=esport, index_name=esindex)
        self.cachedir = cachedir
        self.sim_func = sim_func
        self.timer = Timer(prefix='[timer]')
        self.weighted = weighted
        self.query_terms_only = query_terms_only
        self.base_cache = base_cache

        if not stopwords:
            stopwords = set()
        self._stopwords = stopwords

        if filter_list:
            filter_list = set([e for e in filter_list if e not in stopwords])
        self._filter_list = filter_list

        # calculate figerprint to use as cache comment!
        finger_text = ' '.join([w
                                for w in set.union((self._filter_list
                                                    or set()),
                                                   self._stopwords)])
        finger_md5 = md5()
        finger_md5.update(finger_text.encode('utf-8'))
        self.finger_filter = finger_md5.hexdigest()

    def _in_filter_list(self, elem):
        if elem in self._stopwords:
            return False
        try:
            return (elem in self._filter_list)
        except TypeError:
            return True

    @simple_caching()
    def _get_docs_content_insrank(self, results):
        for res in results:
            content = self.es.get_page(res['id'], res['type'])
            content.pop('references', None)
            res['content'] = [e[1] for e in content.items()]

        return results

    @simple_caching()
    def _tokenize_and_expand(self, qid, question, results):
        """ Takes care of tokenizing the question/results and
            expanding them using one of the four dictionary
            expansion methods.
        """
        cache_comment = self.base_cache + qid
        results = self._get_docs_content_insrank(results,
                                                 cache_comment=cache_comment)
        docs = {r['id']: unicode(r['content']).replace(':', ' ')
                for r in results}
        docs[qid] = question

        # filters out terms
        docs = {did: ' '.join([w for w in doc.split()
                               if self._in_filter_list(w)])
                for did, doc in docs.items()}
        return self.exp_method(docs).objout

    def get_docs(self, qid, question, results):

        cache_comment = (self.base_cache +
                         '{0}_{1}_{2}'.format(qid, self.exp_method.__name__,
                                              self.finger_filter))

        docs = self._tokenize_and_expand(qid,
                                         question,
                                         results,
                                         cache_comment=cache_comment)

        question = Document(qid, 0, docs.pop(qid).split(),
                            float('inf'), 'query',
                            float('inf'), 0)

        doc_results = []
        for res in results:
            res['tokens'] = docs[res['id']].split()

            # eliminates tokens that are not part of the
            # question if specified by argument query_terms_only
            if self.query_terms_only:
                res['tokens'] = [t for t in res['tokens']
                                 if t in question.terms]

            doc_results.append(Document(res['id'], res['rank'], res['tokens'],
                                        res['relevance'], res['type'],
                                        res['score'], res['rank'],
                                        weighted=self.weighted))
        return question, doc_results

    def _swap_position(self, pos_list, posA, posB):
        elA = pos_list[posA]
        elB = pos_list[posB]

        elA.rank = posB + 1
        elB.rank = posA + 1

        pos_list[posB] = elA
        pos_list[posA] = elB

        return True

    def _is_swappable(self, doc, new_rank):
        shift = int(fabs(doc.original_rank - new_rank))
        if shift > self.max_rerank_pos:
            return False
        else:
            return True

    def rerank(self, qid, question, results,
               exp_method, max_rerank_pos=None,
               training_mode=False):
        """ Performs reranking """

        # Retrieves dynamically the methods in expansion_methods
        # using inspect module.
        methods = inspect.getmembers(expansion_methods, inspect.isclass)
        methods = [str(e[1]).split('.') for e in methods]
        methods = [e[len(e) - 1] for e in methods]

        # tries to load such method from expansion_methods. If it
        # fails, it terminates with status 1
        try:
            self.exp_method = getattr(expansion_methods, exp_method)
        except AttributeError:
            print >> sys.stderr, ('[error] {m} is not a valid method: ' +
                                  'use {l}.').format(m=exp_method,
                                                     l=', '.join(methods))
            sys.exit(1)

        # if no maximum number of shifts is set, it lets
        # results move up/down as much as they want
        if not(max_rerank_pos):
            max_rerank_pos = len(results)
        self.max_rerank_pos = max_rerank_pos

        # true if at least a pair of elements have been swapped
        # or if is before first iteration
        swap_flag = True

        self.timer('expansion query {qid}'.format(qid=qid), quiet=True)
        question, docs = self.get_docs(qid, question, results)
        self.timer('expansion query {qid}'.format(qid=qid),
                   quiet=training_mode)

        self.timer('reranking {qid}'.format(qid=qid), quiet=True)
        while swap_flag:
            swap_flag = False
            for (i, j) in [(i, i + 1) for i in range(len(docs) - 1)]:
                sim_i = self.sim_func(question, docs[i])
                sim_j = self.sim_func(question, docs[j])

                if (sim_j > sim_i and
                        self._is_swappable(docs[i], j + 1) and
                        self._is_swappable(docs[j], i + 1)):
                    self._swap_position(docs, i, j)
                    swap_flag = True

        self.timer('reranking {qid}'.format(qid=qid), quiet=training_mode)

        if not training_mode:
            # calculate and print statistics on # of shifts
            rankvals = np.array([fabs(d.original_rank - d.rank)
                                 for d in docs])
            msg = '[info] shift avg: {:.2f}\t shift stdev: {:.2f}'
            print msg.format(rankvals.mean(), rankvals.std())

        out = [{'id': d.id,
                'score': d.score,
                'rank': d.rank,
                'relevance': d.relevance,
                'original_rank': d.original_rank}
               for d in sorted(docs, key=lambda o: o.rank)]

        return out
