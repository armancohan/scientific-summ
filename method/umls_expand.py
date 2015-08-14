#!/usr/bin/python
# -*- coding: utf-8 -*-

import MySQLdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.regexp import RegexpTokenizer
import re
import unicodedata
import evaluate

from annotations_server import documents_model
from method.method_interface import MethodInterface
from util.annotations_client import AnnotationsClient
from util.es_interface import ESInterface
from util.extract_nlptags import Extract_NLP_Tags
from util.metamap14 import run as mmrun
from collections import defaultdict
from util.es_interface import sanitize_string_stringquery as sanitize
import codecs
import json
import os.path
import sys
from util import es_interface
STOPWORDS_PATH = 'data/stopwords.txt'
DOCS_PATH = 'data/TAC_2014_BiomedSumm_Training_Data'
CACHE_FILE = 'cache/umls.json'
with file('data/stopwords-top100.txt') as f:
    stops100 = frozenset([l.strip().lower() for l in f])
RELATIONS = {'alias_of', 'is_a', 'inverse_isa', 'same_as'}
# TODO:
# select STR from MRCONSO where LAT='ENG' and ISPREF='Y' and CUI in (select CUI2 from (select distinct CUI2 from MRREL where CUI1='C
# 0031106' and RELA in ('alias_of', 'is_a','inverse_isa','same_as')) as x);


class Method(MethodInterface):

    """ Produce reference text by submitting the
        citance to the ElasticSearch server.
    """
    method_opts = {'maxsize': {'type': int, 'default': 100},
                   'stopwords-path': {'default': STOPWORDS_PATH},
                   'remove-stopwords': {'default': False,
                                        'action': 'store_true'},
                   'combine': {'default': False, 'action': 'store_true'},
                   'analyzer': {'default': False, 'type': str},
                   'ngram': {'default': False, 'type': int},
                   'runmode': {'default': 'train'}}

    def __init__(self, args, opts):
        super(Method, self).__init__(args, opts)

        self.es_int = ESInterface(host=self.opts.server,
                                  port=self.opts.port,
                                  index_name=self.opts.index_name)
        self.analyzer = self.es_int.get_index_analyzer()
        self.regex_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}(;\s)?)+\s?\)|"
                                         r"(\[(\d+([,–]\s?)?)+\])|"
                                         r"\[[\d,-]+\]").sub
        self.all_digits = re.compile(r"^\d+$").search
        if self.opts.remove_stopwords:
            with file(self.opts.stopwords_path) as f:
                self.stopwords = frozenset([l.strip().lower() for l in f])
        else:
            self.stopwords = frozenset([])
        self.db = MySQLdb.connect(host="127.0.0.1", port=3309,
                                  user="root",
                                  passwd="lollipop11",
                                  db="umls")
        self.cur = self.db.cursor()
        self.ttys = ['SY']

        ttygroups = {"syns": ('AUN', 'EQ', 'SYN', 'MTH'),
                     "chemicals": ('CCN', 'CSN'),
                     "drugs": ('BD', 'BN', 'CD', 'DP', 'FBD', 'GN', 'OCD'),
                     "diseases": ('DI', ), "findings": ('FI', ),
                     "hierarchy": ('HS', 'HT', 'HX'), "related": ('RT', ),
                     "preferred": ('PTN', 'PT')}
        self.doc_mod = documents_model.DocumentsModel(opts.anns_dir)
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

#         if len(args) > 3:s
#             self.ttys = []
#
#             for tty in args[3:]:
#                 if tty in ttygroups:
#                     self.ttys.extend(ttygroups[tty])
#                 else:
#                     self.ttys.append(tty)

    def expand_concept(self, cdata):
        rejected_semTypes = {'ftcn', 'qlco', 'qnco', 'inpr'}
        Okay = True
        for st in cdata['SemanticTypes']:
            if st in rejected_semTypes:
                Okay = False
        if Okay:
            return self.concept_synonyms(cdata['ConceptId'])

    def concept_synonyms(self, cui):
        if cui in evaluate.cachefile:
            return set(evaluate.cachefile[cui])
        else:
            termtypes = ("and (TTY=" +
                         " OR TTY=".join(["'%s'" % x for x in self.ttys]) + ")")
    #         query = 'select * from (select distinct STR from MRCONSO a,'+\
    #                 '(select distinct CUI1,AUI1,AUI2,RELA,CUI2 from MRREL where cui1 = \'%s\'' % cui +\
    #                 ' and rela is not null) b where a.CUI=b.CUI2 and a.LAT=\'ENG\') dd  ;'
            query = "select STR from MRCONSO where " +\
                "CUI = '%s' and LAT = 'ENG' and ISPREF = 'Y'" % cui +\
                termtypes + " and (SAB = 'SNOMEDCT_US')"
#             query = "select STR"
#             print query
            self.cur.execute(query)

#         self.cur.execute("select STR from MRCONSO where " +
#                          "CUI = '%s' and LAT = 'ENG' and ISPREF = 'Y'" % cui +
#                          termtypes + " and SAB != 'CHV'")

            syns = set(filter(lambda y: y.replace(" ", "").isalpha(),
                              [x.lower() for x, in self.cur.fetchall()]))
            evaluate.cachefile[cui] = list(syns)
            return syns

    def umls_expand(self, cui):
        if cui in evaluate.cachefile:
            return set(evaluate.cachefile[cui])
        else:
            termtypes = ("and (TTY=" +
                         " OR TTY=".join(["'%s'" % x for x in self.ttys]) + ")")
    #         query = 'select * from (select distinct STR from MRCONSO a,'+\
    #                 '(select distinct CUI1,AUI1,AUI2,RELA,CUI2 from MRREL where cui1 = \'%s\'' % cui +\
    #                 ' and rela is not null) b where a.CUI=b.CUI2 and a.LAT=\'ENG\') dd  ;'
            query = "select STR from MRCONSO where " +\
                "STR LIKE '%%%s%%' and LAT = 'ENG' and ISPREF = 'Y'" % cui +\
                termtypes + " and (SAB = 'SNOMEDCT_US')"
#             query = "select STR"
#             print query
            self.cur.execute(query)

#         self.cur.execute("select STR from MRCONSO where " +
#                          "CUI = '%s' and LAT = 'ENG' and ISPREF = 'Y'" % cui +
#                          termtypes + " and SAB != 'CHV'")

            syns = set(filter(lambda y: y.replace(" ", "").isalpha(),
                              [x.lower() for x, in self.cur.fetchall()]))
            evaluate.cachefile[cui] = list(syns)
            return syns

    def run(self, test_data):
        out_results = []
        for ann in test_data:
            doc_type = '_'.join((ann['topic_id'].lower(),
                                 ann['reference_article'][:-4].lower()))
            doc_type = doc_type.replace(',', '').replace("'", '"')
            # TEMPORARY FIX FOR WRONG DOCUMENT TYPE NAME
            if self.opts.runmode == 'eval':
                doc_type = doc_type.replace('train', 'eval')

            doc = self.doc_mod.get_doc(
                ann['topic_id'].lower(), ann['citing_article'])
            cit_text = ann['citation_text']
            cit_text_doc = doc[
                ann['citation_offset'][0]:ann['citation_offset'][1]]
            cit_marker = ann['citation_marker']
            cit_marker_doc = doc[
                ann['citation_marker_offset'][0]:ann['citation_marker_offset'][1]]
            cit_mrk_offset_sent = [ann['citation_marker_offset'][0] - ann['citation_offset'][0],
                                   ann['citation_marker_offset'][1] - ann['citation_offset'][0]]
            cleaned = self.reg_apa.sub('', cit_text_doc)
            cleaned = self.reg_ieee.sub('', cleaned)
            cleaned = self.reg_paranthesis.sub('', cleaned)
            cleaned = self.reg_apa_rare.sub('', cleaned)
            cleaned = re.sub('\s+', ' ', cleaned).strip()
            cleaned = re.sub('(,\s)+', ', ', cleaned).strip(', ')

            '''
            -------------- IMMEDIATE NP BEFORE MARKER ----------
            '''
            m = list(self.reg_apa.finditer(cit_text_doc))
            m1 = list(self.reg_ieee.finditer(cit_text_doc))
            m2 = list(self.reg_paranthesis.finditer(cit_text_doc))
            # (start, end, group)
            if len(m) > 0:
                markers = [(e.start(), e.end(), e.group(0)) for e in m]
            elif len(m1) > 0:
                markers = [(e.start(), e.end(), e.group(0))
                           for e in m1]
            elif len(m2) > 0:
                markers = [(e.start(), e.end(), e.group(0))
                           for e in m2]
            else:
                m3 = list(self.reg_apa_rare.finditer(cit_text_doc))
                if len(m3) > 0:
                    markers = [(e.start(), e.end(), e.group(0))
                               for e in m3]
                else:
                    markers = []

            if len(markers) > 10000:

                nps = self.nlp_extractor.parse_by_mbsp(cleaned.strip())
                if nps is None:
                    q = cleaned
                else:
                    t = nps.split(' ')
                    concepts = []
                    for i in range(len(t)):
                        conc = []
                        toks = t[i].split('/')
                        while(('NP' in toks[2]) and (i < len(t))):
                            conc.append((toks[0], toks[6]))
                            i += 1
                            if i < len(t):
                                toks = t[i].split('/')
                        if len(conc) > 0:
                            concepts.append(conc)
                    noun_phrases = [
                        ' '.join([s1[0] for s1 in t1]) for t1 in concepts]

    #                 nps = self.nlp_extractor.extract_NP(cleaned, mode='flattened')
    #                 nps = [[[a[1:-1] for a in piece] for piece in sent] for sent in nps]
        #             nps = [a[1:-1] for sent in nps for piece in sent for a in piece]
    #                 for e in nps:
    #                     noun_phrases = [(sub_e[0].replace('"', ''),idx) for idx, sub_e in enumerate(e) if sub_e[0].replace('"', '') not in self.stopwords]
                    tokens = self.tokenizer.tokenize(cit_text)
                    tokens_offsets = self.tokenizer.span_tokenize(cit_text_doc)
                    nearest = ''
                    nearest_idx = -1
                    distance = 100000
                    # find nearest word to the citation marker
                    for idx, f in enumerate(tokens_offsets):
                        # check to see if in valid span (not citation markers)
                        invalid = False
                        for e in markers:
                            if f[0] >= e[0] and f[1] <= e[1]:
                                invalid = True
                        if (cit_mrk_offset_sent[0] - f[1] >= 0) and\
                                (cit_mrk_offset_sent[0] - f[1] < distance) and\
                                not invalid:
                            distance = cit_mrk_offset_sent[0] - f[1]
                            if len(re.findall(r"^[^A-Za-z]+$", tokens[idx])) == 0:
                                nearest = tokens[idx]
                                if (idx > 0) and len(re.findall(r"^[^A-Za-z]+$", tokens[idx - 1])) == 0:
                                    nearest = tokens[
                                        idx - 1] + ' ' + tokens[idx]
                                nearest_idx = idx
                        elif (cit_mrk_offset_sent[0] < f[1]):
                            break
                        if len(nearest.split(' ')) == 1 and nearest_idx > 0 and\
                                tokens[nearest_idx] not in stops100:
                            nearest = tokens[idx - 1] + ' ' + tokens[idx]
                    largest = 0
                    q = ''
                    for n in noun_phrases:
                        if (nearest in n) and (len(nearest.split()) > largest):
                            q = '"%s"' % nearest
                            largest = len(nearest.split())
                    if q == '':
                        q = cleaned
                q = sanitize(q)
# find longest noun phrase containing the nearest
#                 res = None
#                 for np in nps[0]:
#                    if nearest in np and len(np) > longest and len(np) < 5:
#                        longest = len(np)
#                        res = np
#                 if res is not None:
#                     res = ' '.join([el for el in res])
#                 else:
#                     res = nearest
            else:
                try:
                    qtxt = unicodedata.normalize('NFKD',
                                                 cleaned).encode('ascii', 'ignore')
                except:
                    qtxt = cleaned.encode('ascii', 'ignore')
                qterms = [qtxt]
                qconcepts = mmrun(cleaned)
                for cdata in qconcepts['concepts']:
                    newterms = self.expand_concept(cdata)
                    if newterms is not None:
                        qterms.extend(newterms)

                tokens = self.tokenizer.tokenize(' '.join(qterms))
    #             tokens = self.es_int.tokenize(qtxt, analyzer=self.analyzer)
                q = ' '.join([t for t in tokens
                              if (t not in self.stopwords and
                                  not(self.all_digits(t)))])
                if self.opts.ngram:
                    tokens = self.es_int.tokenize(q, "sentence")
                    new_query = ''
                    for i in range(len(tokens) - self.opts.ngram):
                        tmp = ''
                        for j in range(i, i + self.opts.ngram):
                            tmp += tokens[j] + ' '
                        new_query += '"' + tmp.strip() + '" '
                    q = new_query.strip()

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
#                 beg = e['offset'][0][0] - \
#                     100 if e['offset'][0][0] else e['offset'][0][0]
#                 end = e['offset'][0][1] + 100
#                 e['offset'] = [(beg, end)]
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
                      '_score': sum([e['_score'] for e in r]),
                      'offset': [e['offset'][0] for e in r],
                      'sentence': [e['sentence'] for e in r],
                      '_id': '-000001'}]
            out_results.append(r)
        return out_results
