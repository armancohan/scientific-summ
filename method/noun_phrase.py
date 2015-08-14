#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import codecs
from copy import deepcopy
from method.method_interface import MethodInterface
from util.es_interface import ESInterface
from util.extract_nlptags import Extract_NLP_Tags
from annotations_server import documents_model
from util.annotations_client import AnnotationsClient
STOPWORDS_PATH = 'data/stopwords.txt'
DOCS_PATH = 'data/TAC_2014_BiomedSumm_Training_Data'
from ast import literal_eval


class Method(MethodInterface):

    """ Produce reference text by submitting the
        citance to the ElasticSearch server.
    """
    method_opts = {'maxsize': {'type': int, 'default': 100},
                   'stopwords-path': {'default': STOPWORDS_PATH},
                   'remove-stopwords': {'default': False,
                                        'action': 'store_true'},
                   'remove-stopwords-phrase': {'default': False,
                                               'action': 'store_true'},
                   'noun-phrase': {'default': False,
                                   'action': 'store_true'},
                   'phrase-slop': {'type': int, 'default': 0},
                   'combine': {'default': False, 'action': 'store_true'},
                   'docs-path': {'default': DOCS_PATH},
                   'expand-window': {'default': False, 'action': 'store_true'},
                   'query-terms': {'default': False, 'action': 'store_true'},
                   'verbose': {'default': False, 'action': 'store_true'},
                   'qterm-weight': {'type': float, 'default': 1.0},
                   'phrase-weight': {'type': float, 'default': 2.0},
                   'surrounding-words-weight': {'type': float, 'default': 1.0},
                   'filter-allstops': {'default': False, 'action': 'store_true'},
                   'expand-results': {'type': int, 'default': 0},
                   'sentence': {'default': False, 'type': int},
                   'analyzer': {'default': False, 'type': str}}

    def __init__(self, args, opts):
        super(Method, self).__init__(args, opts)
        self.es_int = ESInterface(host=self.opts.server,
                                  port=self.opts.port,
                                  index_name=self.opts.index_name)
        self.regex_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
                                         r"(\[(\d+([,–]\s?)?)+\])|"
                                         r"\[[\d,-]+\]").sub
        self.all_digits = re.compile(r"^\d+$").search
        if self.opts.remove_stopwords:
            with file(self.opts.stopwords_path) as f:
                self.stopwords = frozenset([l.strip().lower() for l in f])
        else:
            self.stopwords = frozenset([])
        self.doc_mod = documents_model.DocumentsModel(opts.docs_path)
        self.ann_client = AnnotationsClient()

    def run(self, test_data):
        out_results = []
#         outfile = codecs.open('tmp/nlp.txt' , 'wb' , 'UTF-8')
        for ann in test_data:
            doc_type = '_'.join((ann['topic_id'].lower(),
                                 ann['reference_article'][:-4].lower()))
            doc_type = doc_type.replace(',', '').replace("'", '"')

            authors = set((ann['reference_article'][:-4].lower().strip(),
                           ann['citing_article'][:-4].lower().strip()))

            # preprocess (removes citations) and tokenizes
            # citation text before submitting to elasticsearch
            q = self.regex_citation('', ann['citation_text'])
            q = re.sub(r'( ,)+', ',', q)
            q = q.encode('ascii', 'ignore')
            nlp_extractor = Extract_NLP_Tags()
            nps = nlp_extractor.extract_NP(q, mode='flattened')
#             outfile.write('query: "%s" \nparsed: "%s"\n\n' %(q,str(nps)) )
            q1 = ''
            queryterms = set()
            for e in nps:
                for e1 in e:
                    if len(e1) < 4:
                        all_stop = False
                        if self.opts.remove_stopwords_phrase:
                            tmp = ' '.join(sub_e.replace('"', '')
                                           for sub_e in e1 if sub_e.replace('"', '') not in self.stopwords)
                        else:
                            count = 0
                            for sub_e in e1:
                                if sub_e.replace('"', '') in self.stopwords:
                                    count += 1
                            if count == len(e1):
                                all_stop = True
                            tmp = ' '.join(sub_e.replace('"', '')
                                           for sub_e in e1)
                        if tmp not in queryterms and not all_stop:
                            q1 += '"' + tmp + '"^' + \
                                str(self.opts.phrase_weight) + ' '
                            queryterms.add(tmp)
            if self.opts.expand_window:
                window = self.doc_mod.get_para(ann['topic_id'].lower(),
                                               ann['citing_article'][
                                                   :-4].lower(),
                                               (ann['citation_offset'][0],
                                                   ann['citation_offset'][1]))
                sorrounding_text = deepcopy(window['sentence'])
                st = self.regex_citation('', sorrounding_text)
                st = re.sub(r'( ,)+', ',', st)
                st = st.encode('ascii', 'ignore')
                other_nouns = nlp_extractor.extract_NP(st, mode='flattened')
                for e in other_nouns:
                    for e1 in e:
                        if len(e1) < 4:
                            all_stop = False
                            if self.opts.remove_stopwords_phrase:
                                tmp = ' '.join(sub_e.replace('"', '')
                                               for sub_e in e1
                                               if sub_e.replace('"', '') not in self.stopwords)
                            else:
                                count = 0
                                for sub_e in e1:
                                    if sub_e.replace('"', '') in self.stopwords:
                                        count += 1
                                if count == len(e1):
                                    all_stop = True
                                tmp = ' '.join(sub_e.replace('"', '')
                                               for sub_e in e1)
                            if tmp not in queryterms and not all_stop:
                                q1 += '"' + tmp + '"^' + \
                                    str(self.opts.surrounding_words_weight) + \
                                    ' '
                                queryterms.add(tmp)
            if self.opts.query_terms:
                q = ' '.join([t + '^' + str(self.opts.qtrem_weight)
                              for t in self.es_int.tokenize(q)
                              if (t not in self.stopwords and
                                  t not in authors and
                                  not(self.all_digits(t)))])
                q1 = q1 + ' ' + q
            if self.opts.verbose:
                print "query:   %s" % q
                print "q1   :       %s" % q1
                print '_____'
#             q2 = self.es_int.tokenize(q1, 'sentence')
#             q2 = ' '.join([t for t in self.es_int.tokenize(q1)
#                           if (t not in self.stopwords and
#                               t not in authors and
#                               not(self.all_digits(t)))])
            if self.opts.analyzer:
                r = self.es_int.simple_search(q1.strip(), maxsize=self.opts.maxsize,
                                              source_fields=[
                                                  'offset', 'sentence'],
                                              # field='sentence',
                                              doc_type=doc_type,
                                              phrase_slop=self.opts.phrase_slop,
                                              params={'analyzer': self.opts.analyzer})
            else:
                r = self.es_int.simple_search(q1.strip(), maxsize=self.opts.maxsize,
                                              source_fields=[
                                                  'offset', 'sentence'],
                                              # field='sentence',
                                              doc_type=doc_type,
                                              phrase_slop=self.opts.phrase_slop)
            if self.opts.sentence:
                for idx, e in enumerate(deepcopy(r)):
                    if '_id' in e:
                        query = ' OR '.join(['_id:%s'%(str(int(e['_id']) + j).zfill(5))
                                           for j
                                           in range(-1*self.opts.sentence, self.opts.sentence+1) if j!=0 and int(e['_id'])+j > 0])
                        sour = self.es_int.simple_search(query, doc_type=e['_type'], maxsize=2*self.opts.sentence, source_fields=['offset', 'sentence'])
#                         aft = self.es_int.get_page(
#                             str(int(e['_id']) + 1).zfill(5), e['_type'])
#                         bef = self.es_int.get_page(
#                             str(int(e['_id']) + 1).zfill(5), e['_type'])
                        if len(sour) > 0:
                            for s in sour:
                                r.insert(idx + 1, s)

            for e in r:
                fld = e.pop('fields')
                if eval(fld['offset'][0])[0] < self.opts.expand_results:
                    beg = 0
                else:
                    beg = eval(fld['offset'][0])[0] - self.opts.expand_results
                endd = eval(fld['offset'][0])[1] + self.opts.expand_results
                e['offset'] = [(beg, endd)]
                e['sentence'] = fld['sentence'][0]
                e['query'] = q1

            r1 = deepcopy(r)
            r = []
            for idx, e in enumerate(r1):
                if idx < self.opts.maxsize:
                    r.append(e)

            if self.opts.combine:
                if len(r) == 0:
                    r = [{'_type': doc_type,
                          '_index': self.opts.index_name,
                          '_score': 0,
                          'sentence': '',
                          'offset': [(0, 1)],
                          'query':q1, '_id':-11}]
                r = [{'_type': r[0]['_type'],
                      '_index': r[0]['_index'],
                      'query': q1,
                      '_score': sum([e['_score'] for e in r]),
                      'offset': [e['offset'][0] for e in r],
                      'sentence': [e['sentence'] for e in r],
                      '_id': '-000001'}]
            out_results.append(r)
        return out_results
