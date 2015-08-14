from __future__ import division
from util.tokenization import SentTokenizer, WordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from base import Summarizer
import numpy as np
from heapq import heappush, heappop, heappushpop
import codecs
import json
from nltk.corpus import stopwords


class MMR(Summarizer):
    '''
    classdocs
    '''

    def __init__(self, lmbda=0.3):
        '''
        Initializes MMR

        Args:
            lmbda(float) the lambda param for MMR
        '''
        self.lmbda = lmbda
        self.w_t = WordTokenizer(stem=False)

    def summarize(self, doc, max_length=10):
        '''
        Summarizes a document or list of docs
        MMR(S) = lambda*Sim(S,D)-(1-lambda)*Sim(S,Summary)
        Arg:
            doc: (list) | (str)

            max_length: The maximum length of the desired summary

        Returns
        -------
        str
        '''
        if isinstance(doc, str):  # list of sentences, no need to tokenize
            s_t = SentTokenizer()
            docs = s_t(doc, offsets=False)
            docs += [doc]  # Dummy sentence, The whole document
        else:
            docs = doc + [' '.join(doc)]
        tokzr = self.get_tokenizer('regex', True)
        vectorizer = TfidfVectorizer(
            min_df=1, max_df=len(doc) * .95,
            tokenizer=tokzr,
            stop_words=stopwords.words('english'))
        vectors = vectorizer.fit_transform(docs).toarray()
        doc_texts = {i: v for i, v in enumerate(docs)}
        doc_dict = {i: v for i, v in enumerate(vectors)}
        feature_names = vectorizer.get_feature_names()
#         idf_vals = vectorizer.idf_

        summ_scores = []  # includes tuples (mmr_score, sentence_id)
        for i, s in doc_texts.iteritems():  # iterate through sentences to
                                            # select them for summary
            if len(summ_scores) > 0:
                summ_v = ' '.join([doc_texts[e[1]] for e in summ_scores])
                # summarization vector
            else:
                summ_v = ''
            if summ_v != '':
                summ_v = vectorizer.transform(
                    [summ_v]).toarray()[0]  # to tf-idf
            score = -1 * self._mmr(
                vectorizer.transform(
                    [s]).toarray()[0], doc_dict[len(doc_dict) - 1],
                summ_v, self.lmbda, self.cossim)
            if len(summ_scores) < max_length / 30 + 3:
                # max heap data structure for mmr
                heappush(summ_scores, (score, i))
            else:  # Get rid of lowest score
                heappushpop(summ_scores, (score, i))
        print summ_scores
        final_sum = []
        for s in summ_scores:
            if self.w_t.count_words(final_sum) < max_length:
                final_sum.append(doc_texts[s[1]])
#         print 'before: %d' % self.w_t.count_words(final_sum)
        if self.w_t.count_words(final_sum) > max_length:
            tmp = final_sum.pop()
        if self.w_t.count_words(final_sum) == 0:
            final_sum.append(tmp)
#         print 'after: %d' % self.w_t.count_words(final_sum)
        return final_sum

    def _mmr(self, s, D, Summ, lmbda, sim):
        '''
        s: Sentence for evaluation
        D: The whole document
        Summ: The summary
        lmbda: Lambda parameter
        sim: The similarity function

        Returns
        ------
        float
        '''
        if Summ == '':
            return lmbda * sim(s, D)
        return lmbda * sim(s, D) - (1 - lmbda) * (sim(s, Summ))

