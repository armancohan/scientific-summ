import hashlib
import shelve
import json
from sys import stderr

from elasticsearch import Elasticsearch as ES
from elasticsearch.client.indices import IndicesClient
from elasticsearch.exceptions import RequestError, TransportError

import requests
import math
import constants
import sys

SIM_MODELS_NAMES = ['LMDirichlet', 'LMJelinekMercer',
                    'IB', 'BM25', 'default', 'DFR']

TO_ESCAPE = ['+', '-', '&&', '||', '!', '(', ')', '{', '}', '[',
             ']', '^', '"', '~', '*', '?', ':', '/']


class ESInterface():

    """Interface for ElasticSearch"""
    _count_total = -1  # N: Number of docs
    _idf = None  # dict for storing idf values

    def __init__(self, host='localhost', port=9200,
                 index_name='biosum', cred_path='.cred'):
        self.host = host
        self.port = port
        self.index_name = index_name
        self.cred_path = cred_path
        # self.doc_type = 'papers'
        self.es = self.__connect()
        self.ic = IndicesClient(self.es)
        try:
            cache_file = constants.get_path()['cache']
            self.page_cache = shelve.open(cache_file + '/pages.p',
                                          writeback=False)
        except:
            print 'Not found: %s' % cache_file
            print sys.exc_info()[0]
            sys.exit()

    def login(self, username, password):
        pass

    @property
    def description(self):
        # get mapping, clean it up
        m = self.es.indices.get_mapping(self.index_name)
        m = m[self.index_name]['mappings']

        description = {'host': self.host,
                       'port': self.port,
                       'index_name': self.index_name,
                       'mapping': m}
        return description

    @property
    def size(self):
        stats = self.es.indices.stats()['indices'][self.index_name]
        return stats['total']['docs']['count']

    def __connect(self):
        '''Private method used to connect to the ElasticSearch instance.'''
        es = ES(hosts=[{'host': self.host, 'port': self.port}])

        # checks if server exists
        if not es.ping():
            err = ('It appears that nothing is running at http://%s:%s' %
                   (self.host, self.port))
            raise OSError(err)

        # load the credentials file (if possible)
#         with file(self.cred_path) as cf:
#             username, password = [l.strip() for l in cf.readlines()][:2]
#         data = json.dumps({'username': username, 'password': password})
        url = 'http://%s:%s/login' % (self.host, self.port)
        resp = json.loads(requests.post(url).text)
#         if resp['status'] == 200:
#             self.auth_token = resp['token']
#         else:
#             self.auth_token = ''

        # checks if index exists
        try:
            es.indices.get_mapping(self.index_name)
        except TransportError as e:
            if e.args[0] == 403:
                err = list(e.args)
                err[1] = ('Credentials not valid for %s:%s/%s' %
                          (self.host, self.port, self.index_name))
                e.args = tuple(err)
            elif e.args[0] == 404:
                self.__del__()
                err = list(e.args)
                err[1] = ('No index named "%s" is avaliable at %s:%s' %
                          (self.index_name, self.host, self.port))
                e.args = tuple(err)
            raise
        return es

    def __del__(self):
        requests.post('http://%s:%s/logout' % (self.host, self.port))

    # def get_scroll(self, scroll_size, scroll_timeout):
    #     q_body = {"query": {"match_all": {}}}
    #     return self.es.search(self.index_name, self.doc_type, q_body,
    #                           search_type='scan', scroll='100m',
    #                           size='10000')

    # def scroll(self, scroll_id):
    #     return self.es.scroll(scroll_id, scroll='10m')

    # def scan_and_scroll(self, doc_type, scroll_size=50, scroll_timeout=10):
    #     """
    #     The scan search type allows to efficiently scroll a large result set.
    #     The response will include no hits, with two important results,
    #     the total_hits will include the total hits that match the query
    #     and the scroll_id that allows to start the scroll process.

    #     @param scroll_size: scroll size
    #     @param scroll_timeout: rountdtrip timeout
    #     """
    #     q_body = {"query": {
    #         "match_all": {}
    #     }}
    #     result = self.es.search(self.index_name,
    #                             doc_type,
    #                             q_body,
    #                             search_type='scan',
    #                             scroll=str(scroll_timeout) +
    #                             'm',
    #                             size=scroll_size)
    #     res = self.es.scroll(
    #         result['_scroll_id'], scroll=str(scroll_timeout) + 'm')
    #     finalres = []
    #     while len(res['hits']['hits']) > 0:
    #         finalres.append(res)
    #         res = self.es.scroll(
    #             res['_scroll_id'], scroll=str(scroll_timeout) + 'm')
    #     return finalres

    # def esc(self, txt):
    #     for e in TO_ESCAPE:
    #         txt = txt.replace(e, '\%s' % e)
    #     return txt

    def find_all(self, source_fields=None, doc_type=''):
        if source_fields:
            q_body = {
                "fields": source_fields,
                "query": {
                    "match_all": {}
                }
            }
        else:
            q_body = {
                "query": {
                    "match_all": {}
                }
            }
        return self.es.search(
            body=q_body, size=1000000, index=self.index_name, doc_type=doc_type)['hits']['hits']

    def multi_field_search(self,
                           field_vals,
                           fields=['sentence', 'mm-concepts', 'noun_phrases'],
                           maxsize=1000,
                           field_boost=[1, 3, 2],
                           offset=0,
                           source_fields=[],
                           doc_type='',
                           params=None):
        '''Interface for simple query tasks.
        Parameters:
            - field_vals [requried]: a list of field values to query
            - maxsize [optional]:   number of results to get.
                                    default is 1000.
        Returns results.'''
#         q_body = {
#             "fields": source_fields,
#             "query": {
#                 "dis_max": {
#                     "queries": [
#                         {"match": {
#                             "sentence":  {
#                                 "query": sentence,
#                                 "boost": field_boost[0]
#                             }}},
#                         {"match": {
#                             "mm-concepts":  {
#                                 "query": concepts,
#                                 "boost": field_boost[1]
#                             }}},
#                         {"match": {
#                             "noun_phrases":  {
#                                 "query": noun_phrases,
#                                 "boost": field_boost[2]
#                             }}}
#                     ]
#                 }
#             }
#         }
        q_body = {
            "fields": source_fields,
            "query": {
                "dis_max": {
                    "queries": [
                    ]
                }
            }
        }
        for idx in range(len(field_vals)):
            q_body['query']['dis_max']['queries'].append({"match": {
                fields[idx]:  {
                    "query": field_vals[idx],
                    "boost": field_boost[idx]
                }}})

        if params is not None:
            for key in params:
                q_body['query']['dis_max'][key] = params[key]

        return self._cursor_search(q_body, maxsize, offset, doc_type)

    def simple_search(self, query, field='_all', maxsize=1000,
                      offset=0, source_fields=[], doc_type='',
                      operator='or', phrase_slop=0, escape=False, params=None):
        '''Interface for simple query tasks.
        Parameters:
            - query [requried]: the string to query
            - maxsize [optional]:   number of results to get.
                                    default is 1000.
        Returns results.'''

        if escape:
            query = self.esc(query)

        q_body = {
            "fields": source_fields,
            'query': {
                'query_string': {
                    'query': query,
                    'default_operator': operator,
                    'use_dis_max': True,
                    'auto_generate_phrase_queries': True,
                    'phrase_slop': phrase_slop
                }
            }
        }
        if params is not None:
            for key in params:
                q_body['query']['query_string'][key] = params[key]

        if field:
            q_body['query']['query_string']['default_field'] = field

        return self._cursor_search(q_body, maxsize, offset, doc_type)

    def count(self, query, field='_all', operator="AND"):
        q = {
            'query': {
                "query_string": {
                    "default_field": field,
                    "default_operator": operator,
                    "query": query
                }
            }
        }
        resp = self.es.count(body=q, index=self.index_name)
        if resp['_shards']['failed'] > 0:
            raise RuntimeError("ES count failed: %s", resp)

        return resp['count']

    def _cursor_search(self, q, maxsize, offset, doc_type):
        return self.es.search(index=self.index_name,
                              body=q,
                              size=maxsize,
                              from_=offset,
                              doc_type=doc_type)['hits']['hits']

    def update_field(self, docid, doc_type,
                     field_name, field_value):
        ''' Update field field_name with field_value'''
        body = {'doc': {field_name: field_value}}
        self.es.update(id=docid, doc_type=doc_type,
                       index=self.index_name, body=body)

    def get_page_by_res(self, res_dict, cache=False):
        return self.get_page(res_dict['_id'],
                             res_dict['_type'],
                             cache=cache)

    def get_page(self, docid, doc_type, cache=False):
        ''' Retrieve a page's source from the index
        Parameters:
            - id [required]: the ES id of the page to retrieve
            - doc_type [required]: the ES document type to retrieve
        '''
        k = str("-".join((docid, self.index_name, doc_type)))

        if not cache or k not in self.page_cache:
            page = self.es.get_source(id=docid,
                                      index=self.index_name,
                                      doc_type=doc_type)

            if cache:
                self.page_cache[k] = page
                self.page_cache.sync()
        else:
            page = self.page_cache[k]

        return page

    def get_index_analyzer(self):
        return self.ic.get_settings(index=self.index_name)\
            [self.index_name]['settings']['index']\
            ['analysis']['analyzer'].keys()[0]

    def tokenize(self, text, field="text", analyzer=None):
        ''' Return a list of tokenized tokens
        Parameters:
            - text [required]: the text to tokenize
            - field [optional]: the field whose ES analyzer
                                should be used (default: text)
        '''
        params = {}
        if analyzer is not None:
            params['analyzer'] = analyzer
        try:
            response = self.ic.analyze(body=text, field=field,
                                       index=self.index_name,
                                       params=params
                                       )
            return [d['token'] for d in response['tokens']]
        except RequestError:
            return []

    def phrase_search(self, phrase, doc_type='',
                      field='_all', slop=0, in_order=True,
                      maxsize=1000, offset=0, source_fields=[]):
        ''' Retrieve documents containing a phrase.
            Does not return the documents' source. '''

        phraseterms = self.tokenize(phrase, field=field)
        if len(phraseterms) == 0:
            return []

        q = {
            "fields": source_fields,
            "query": {
                "span_near": {
                    "clauses": [{"span_term": {field: term}}
                                for term in phraseterms],
                    "slop": slop,  # max number of intervening unmatched pos.
                    "in_order": in_order,
                    "collect_payloads": False
                }
            }
        }
        return self._cursor_search(q, maxsize, offset, doc_type)

    def phrase_count(self, phrase, field='_all', slop=0, in_order=True):
        phraseterms = self.tokenize(phrase, field=field)

        if len(phraseterms) == 0:
            return []

        q = {
            "query": {
                "span_near": {
                    "clauses": [{"span_term": {field: term}}
                                for term in phraseterms],
                    "slop": slop,  # max number of intervening unmatched pos.
                    "in_order": in_order,
                    "collect_payloads": False
                }
            }
        }

        resp = self.es.count(body=q, index=self.index_name)
        if resp['_shards']['failed'] > 0:
            raise RuntimeError("ES count failed: %s", resp)

        return resp['count']

    def index_hash(self):
        ''' Weak hash (only considers mapping and size) of index_name '''
        ic_sts = self.ic.stats(index=self.index_name)['_all']['total']['store']
        ic_map = self.ic.get_mapping(index=self.index_name)
        s = "_".join((unicode(ic_sts), unicode(ic_map)))
        return hashlib.md5(s).hexdigest()

    # def get_mappings(self):
    #     mappings = self.es.indices.get_mapping(self.index_name)
    #     return mappings[self.index_name]['mappings']

    def set_mappings(self, mapdict):
        ''' Set mapping for documents in index according to map_dict;
            only documents types with an entry in map dict are updated.
            No input check; PLEASE FOLLOW SPECIFICATIONS!
            format:
            {<doc_type_1>: {'properties': {'doc_field_1': {<properties>}
                                           ...
                                           'doc_filed_n': {<properties>}
                                           }
                            }
            }
        '''
        for doc_type, mapping in mapdict:
            self.es.indices.put_mapping(index=self.index_name,
                                        doc_type=doc_type,
                                        body=mapping)

    # def get_ids(self, doc_type):
    #     res = self.scan_and_scroll(doc_type, scroll_size=5000)
    #     return res

    # def get_types(self):
    #     from subprocess import check_output
    #     request = 'http://localhost:9200/indexname/_mapping?pretty=1'
    #     request = request.replace('indexname', self.index_name)
    #     res = json.loads(check_output(["curl", "-XGET", request]))
    #     return res[self.index_name]['mappings'].keys()

    def get_termvector(self, doc_type, docid, fields=None):
        """ Return the term vector and stratistics
            for document docid of type doc_type.
            If fields is not provided, term vectors
            are returned for each field.
        """
        if fields is None:
            fields = []
        body = {
            "fields": fields,
            "offsets": True,
            "payloads": True,
            "positions": True,
            "term_statistics": True,
            "field_statistics": True
        }
        resp = self.es.termvector(index=self.index_name,
                                  doc_type=doc_type,
                                  id=docid,
                                  body=body)
        return resp

    def add(self, index,
            doc_type,
            entry,
            docid=None):
        self.es.index(index=index, doc_type=doc_type, body=entry,
                      id=docid)

    def get_avg_size(self, field):
        '''
        Get the average document length for a the field sentence
        '''
        q = {"fields": [
            "sentence"
        ],
            "query": {
            "match_all": {

            }
        },
            "aggs": {
            "my_agg": {
                "avg": {
                    "script": "doc['sentence'].size()"
                }
            }
        }
        }
        res = self.es.search(index=self.index_name, body=q)
        return res['aggregations']['my_agg']['value']

    def get_idf(self, term):
        '''
        Returns the idf of a given term on the index

        Args:
            term(str)

        Returns:
            float -- idf value
        '''
        if self._count_total == -1:
            self._count_total = self.count(query='*:*')
        if self._idf is not None:
            if term in self._idf:
                return self._idf[term]
            else:
                count = self.count(term)
                if count == 0:
                    idf = 0
                else:
                    idf = math.log(
                        (self._count_total - count + 0.5) / (count + 0.5))
                self._idf[term] = idf
        else:
            count = self.count(term)
            if count == 0:
                idf = 0
            else:
                idf = math.log(
                    (self._count_total - count + 0.5) / (count + 0.5))
            self._idf = {term: idf}
        return idf

    def get_types(self):
        '''
        Get all types that are under the index (self.index_name)
        '''
        idx_mapping = self.ic.get_mapping(index=self.index_name)
        return idx_mapping.values()[0]['mappings'].keys()

    def get_all_ids(self, doc_type):
        hits = self._cursor_search(q={"fields": ["_id"], "query": {"match_all": {}}},
                                   maxsize=100000, offset=0, doc_type=doc_type)
        ids = [e['_id'] for e in hits]
        return sorted(ids)

    def get_all_termvectors(self, doc_type, field='sentence'):
        '''
        Returns all termvectors for a given doc_type

        Args:
            doc_type(str)
            field(str): The field for which we want the term vectors

        Returns:
            dict
                key: term
                value: term_stats
        '''
        ids = self.get_all_ids(doc_type)
        tv = {}  # term_vectors
        for id in ids:
            t = self.get_termvector(doc_type, docid=id, fields=['sentence'])
            for term in t['term_vectors'][field]['terms'].keys():
                tv[term] = t['term_vectors'][field]['terms'][term]
        return tv

import re


def sanitize_string_stringquery(input_txt):
    s = re.escape('\\+-&|!(){}[]^~*?:/')
    return re.sub(r'([' + s + '])', '\\\\\g<1>', input_txt)
