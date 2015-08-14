
from __future__ import print_function
import json
import sys
import requests

_LOGIN_INFO = '.creds'
"""
Credentials file format:
{'username':'user',
'password':'password'}
"""


class AnnotationsClient(object):

    def __init__(self, credentials=None, host='localhost', port=3003):
        print(_LOGIN_INFO)
        if credentials is None:
            try:
                with file(_LOGIN_INFO) as cred_file:
                    credentials = json.load(cred_file)
            except (IOError) as e:
                e.args = ['No credentials provided.']
                raise
            except (ValueError) as e:
                raise
        username = credentials['username']
        password = credentials['password']

        self.endpoint = 'http://%s:%s' % (host, port)
        if not self.ping():
            raise requests.ConnectionError('Nothing appears to be running at '
                                           '%s' % self.endpoint)

        success = self._connect(username, password)
        if not success:
            raise requests.ConnectionError('Username/password combination '
                                           'rejected.')

    def ping(self, endpoint=None):
        if endpoint is None:
            endpoint = self.endpoint
        try:
            req = requests.get(self.endpoint)
            return json.loads(req.content)['ping']
        except requests.ConnectionError:
            return False

    def _connect(self, username, password):
        token = requests.post('%s/%s' % (self.endpoint, 'login'),
                              auth=(username, password))
        if token:
            self.token = token.content
            return True
        else:
            return False

    def get_doc_from_es(self, result):
        topic_id = '_'.join(result['_type'].split('_')[:2])
        doc_name = result['_type'].split('_')[2]
        return self.get_doc(topic_id, doc_name, result['offset'][0])

    def get_doc(self, topic_id, doc_name, offset=None):
        params = {'topic_id': topic_id, 'doc_name': doc_name}
        if offset:
            if type(offset) not in (list, tuple):
                raise Exception('offset malformatted: is %s, '
                                'should be tuple or list.' % type(offset))
            else:
                params.update({'start': offset[0], 'end': offset[1]})

        resp = requests.get('%s/%s' % (self.endpoint, 'document'),
                            headers={'Authorization': self.token},
                            params=params)

        if resp.status_code != 404:
            return json.loads(resp.content)
        else:
            print(('[error] document %s/%s was not found.'
                   '' % (topic_id, doc_name)),
                  file=sys.stderr)
            return None

    def get_para(self, topic_id, doc_name, offset):
        params = {'topic_id': topic_id.lower(),
                  'doc_name': doc_name.lower().replace('.txt', '')}
        if type(offset) not in (list, tuple):
            raise Exception('offset malformatted: is %s, '
                            'should be tuple or list.' % type(offset))
        else:
            params.update({'start': offset[0], 'end': offset[1]})

        resp = requests.get('%s/%s' % (self.endpoint, 'paragraph'),
                            headers={'Authorization': self.token},
                            params=params)

        if resp.status_code != 404:
            return resp.content
        else:
            print(('[error] document %s/%s was not found.'
                   '' % (topic_id, doc_name)),
                  file=sys.stderr)
            return None

if __name__ == '__main__':
    print('### TEST ###')
    ac = AnnotationsClient()
    r = (u'D1410_TRAIN', u'Du,Hu.txt', (12092, 12508))
    resp = ac.get_para(*r)
    print(resp)
    print('\n')
    resp = ac.get_doc(*r)
    print(resp)
