#!/usr/bin/python
# -*- coding: utf-8 -*-

from ast import literal_eval
import codecs
from copy import deepcopy
from nltk.tokenize.regexp import RegexpTokenizer
import re

from annotations_server import documents_model
from method.method_interface import MethodInterface
from util.annotations_client import AnnotationsClient
from util.es_interface import ESInterface
from util.extract_nlptags import Extract_NLP_Tags
from nltk.stem.wordnet import WordNetLemmatizer
from timeit import itertools
STOPWORDS_PATH = 'data/stopwords.txt'
DOCS_PATH = 'data/TAC_2014_BiomedSumm_Training_Data'


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

        self.reg_apa = re.compile(
            # [Chen et al.2000]
            r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
            r"\(\s?([^ ]+\s?[^ ]*et\sal\.,?\s\d{2,4}(,\s)?)+(\sand\s)?[^ ]+\s?[^ ]*et\sal\.,?\s\d{2,4}\)|"
            r"\w+\set al\. \(\d{2,4}\)")  # [Chen et al. 200]
        self.reg_apa_rare = re.compile(
            r"((([A-Z]\w+\set\sal\.,? \d{4})|([A-Z]\w+\sand\s[A-Z]\w+,?\s\d{4}))((,\s)| and )?)+")
        self.reg_apa2 = re.compile(
            r"\(\s?(\w+\s?\w*et\sal\.,\s\d{2,4}(,\s)?)+(\sand\s)?\w+\s?\w*et\sal\.,\s\d{2,4}\)")
        self.reg_ieee = re.compile(r"(\[(\d+([,–]\s?)?)+\])|\[\s?[\d,-]+\]")
        self.reg_paranthesis = re.compile(
            r"\(\s?\d{1,2}(,\s\d{1,2})*(\sand\s\d{1,2})?\)")
        self.nlp_extractor = Extract_NLP_Tags()
        self.tokenizer = RegexpTokenizer('[^\w\-\']+', gaps=True)
        self.lmtzr = WordNetLemmatizer()

    def run(self, test_data):
        out_results = []
        not_found = 0
        total = 0
#         outfile = codecs.open('tmp/nlp.txt' , 'wb' , 'UTF-8')
        processed = set()
        for ann in test_data:
            if (ann['topic_id'] + '_' + str(ann['citance_number'])) not in processed:
                doc_type = '_'.join((ann['topic_id'].lower(),
                                     ann['reference_article'][:-4].lower()))
                doc_type = doc_type.replace(',', '').replace("'", '"')
                doc = self.doc_mod.get_doc(
                    ann['topic_id'].lower(), ann['citing_article'])
                cit_text = ann['citation_text']
                cit_text_doc = doc[
                    ann['citation_offset'][0]:ann['citation_offset'][1]]
                cit_marker = ann['citation_marker']
                cit_marker_doc = doc[
                    ann['citation_marker_offset'][0]:ann['citation_marker_offset'][1]]
                cit_mrk_offset_sent = [ann['citation_marker_offset'][0] - ann['citation_offset'][0] + 1,
                                       [ann['citation_marker_offset'][1] - ann['citation_offset'][1] + 1]]
                cleaned = self.reg_apa.sub('', cit_text_doc)
                cleaned = self.reg_ieee.sub('', cleaned)
                cleaned = self.reg_paranthesis.sub('', cleaned)
                cleaned = self.reg_apa_rare.sub('', cleaned)
                cleaned = re.sub('\s+', ' ', cleaned).strip()
                cleaned = re.sub('(,\s)+', ', ', cleaned).strip(', ')
                chunks = set()
                # get noun phrases, format [[[term1, term2],[term3]][term4,
                # term5]]
                nps = self.nlp_extractor.extract_NP(cleaned, mode='flattened')
#                 nps = [[[a[1:-1] for a in piece] for piece in sent] for sent in nps]
#                 for e in nps:
#                     noun_phrases = [(sub_e[0].replace('"', ''),idx) for idx, sub_e in enumerate(e) if sub_e[0].replace('"', '') not in self.stopwords]
                noun_phrases = [e for e in list(itertools.chain.from_iterable(nps))
                                if e not in self.stopwords]
#                 tokens = self.tokenizer.tokenize(cit_text)
#                 tokens_offsets = self.tokenizer.span_tokenize(cit_text_doc)
#                 cleaned = ''
#
#                 m = list(self.reg_apa.finditer(cit_text_doc))
#                 m1 = list(self.reg_ieee.finditer(cit_text_doc))
#                 m2 = list(self.reg_paranthesis.finditer(cit_text_doc))
#                 # (start, end, group)
#                 if len(m) > 0:
#                     markers = [(e.start(), e.end(), e.group(0)) for e in m]
#                 elif len(m1) > 0:
#                     markers = [(e.start(), e.end(), e.group(0))
#                                for e in m1]
#                 elif len(m2) > 0:
#                     markers = [(e.start(), e.end(), e.group(0))
#                                for e in m2]
#                 else:
#                     m3 = list(self.reg_apa_rare.finditer(cit_text_doc))
#                     if len(m3) > 0:
#                         markers = [(e.start(), e.end(), e.group(0))
#                                    for e in m3]
#                     else:
#                         not_found += 1
#                 nearest = ''
#                 distance = 100000
#                 if len(markers) > 1:
#                     # find nearest word to the citation marker
#                     for idx, f in enumerate(tokens_offsets):
#                         # check to see if in valid span (not citation markers)
#                         invalid = False
#                         for e in markers:
#                             if f[0] >= e[0] and f[1] <= e[1]:
#                                 invalid = True
#                         if (cit_mrk_offset_sent[0] - f[1] >= 0) and\
#                                 (cit_mrk_offset_sent[0] - f[1] < distance) and\
#                                 not invalid:
#                             distance = cit_mrk_offset_sent[0] - f[1]
#                             if len(re.findall(r"^[^A-Za-z]+$", tokens[idx])) == 0:
#                                 nearest = tokens[idx]
#
#                         # find longest noun phrase containing the nearest
#                         longest = 0
#                         res = None
#                         for np in nps[0]:
#                             if nearest in np and len(np) > longest:
#                                 longest = len(np)
#                                 res = np
#                         if res is not None:
#                             res = ' '.join([el for el in res])
#                         else:
#                             res = nearest
#                 else:
#                     # if there is only one citation marker, just consider the
#                     # whole citation text as the query
#                     q_tokens = []
#                     for idx, f in enumerate(tokens_offsets):
#                         invalid = False
#                         for e in markers:
#                             if f[0] >= e[0] and f[1] <= e[1]:
#                                 invalid = True
#                         if (cit_mrk_offset_sent[0] - f[1] >= 0) and\
#                                 (cit_mrk_offset_sent[0] - f[1] < distance) and\
#                                 not invalid:
#                             q_tokens.append(tokens[idx])
#                     res = ' '.join([f for f in q_tokens])
                q = noun_phrases
                q = ' '.join(q).encode('ascii', 'ignore')
    #             outfile.write('query: "%s" \nparsed: "%s"\n\n' %(q,str(nps)) )
                tokens = self.es_int.tokenize(q, "sentence")
                q = ' '.join([t for t in tokens
                              if (t not in self.stopwords and
                                  not(self.all_digits(t)))])
                if self.opts.analyzer:
                    r = self.es_int.simple_search(q, maxsize=self.opts.maxsize,
                                                  source_fields=[
                                                      'offset', 'sentence'],
                                                  # field='sentence',
                                                  doc_type=doc_type,
                                                  params={'analyzer': self.opts.analyzer})
                else:
                    r = self.es_int.simple_search(q, maxsize=self.opts.maxsize,
                                                  source_fields=[
                                                      'offset', 'sentence'],
                                                  # field='sentence',
                                                  doc_type=doc_type)
                for e in r:
                    fld = e.pop('fields')
                    e['offset'] = [eval(fld['offset'][0])]
                    beg = e['offset'][0][0] - \
                        100 if e['offset'][0][0] else e['offset'][0][0]
                    end = e['offset'][0][1] + 100
                    e['offset'] = [(beg, end)]
                    e['sentence'] = fld['sentence'][0]
                    e['query'] = q
                if self.opts.combine:
                    if len(r) == 0:
                        r = [{'_type': doc_type,
                              '_index': self.opts.index_name,
                              '_score': 0,
                              'sentence': '',
                              'offset': [(0, 1)],
                              'query':q, '_id':-11}]
                    r = [{'_type': r[0]['_type'],
                          '_index': r[0]['_index'],
                          'query': q,
                          'topic': ann['topic_id'].lower(),
                          'citance_number': ann['citance_number'],
                          'citation_text': ann['citation_text'],
                          'citing_article': ann['citing_article'],
                          '_score': sum([e['_score'] for e in r]),
                          'offset': [e['offset'][0] for e in r],
                          'sentence': [e['sentence'] for e in r],
                          '_id': '-000001'}]
                out_results.append(r)
        return out_results

#             q2 = self.es_int.tokenize(q1, 'sentence')
#             q2 = ' '.join([t for t in self.es_int.tokenize(q1)
#                           if (t not in self.stopwords and
#                               t not in authors and
#                               not(self.all_digits(t)))])
