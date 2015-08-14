from flask import Flask, request, jsonify, Response
from documents_model import DocumentsModel
from optparse import OptionParser
import json
from os import urandom
from hashlib import md5, sha256
from functools import wraps
import constants

"""
Expects a credential file with name .login at annotations_server/ path
The format of the file:
{"username":"password"}
"""

op = OptionParser()
op.add_option('-H', '--host', dest='host', default=constants.flask_server)
op.add_option('-P', '--port', dest='port', default=constants.flask_port, type=int)
op.add_option('-a', '--ann-path', dest='ann_path',
              default=constants.flask_data)
op.add_option('--debug', default=False, action='store_true')
op.add_option('--cred-path', default=constants.flask_server_login)

opts, args = op.parse_args()

app = Flask(__name__)
dm = DocumentsModel(opts.ann_path, verbose=opts.debug)
app.debug = opts.debug
tokens_dict = {}
tokens_set = set([])

seed = urandom(256)

with file(opts.cred_path) as fls:
    users = json.load(fls)
    print users
if opts.debug:
    print '%s user loaded' % len(users)

def hash_obj(o):
    h = sha256(seed)
    h.update(str(o))
    return h.hexdigest()


def token_check(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token in tokens_set or opts.debug:
            return method(*args, **kwargs)
        else:
            return Response('Not authorized.', 401)
    return wrapper


@app.route('/', methods=['GET'])
def is_running():
    return(jsonify(ping=True))


@app.route('/login', methods=['POST'])
def login():
    login = request.authorization
    if (login['username'] in users and
            users[login['username']] == login['password']):
        h = hash_obj(login)
        r = tokens_dict.setdefault(h, md5(urandom(2048)).hexdigest())
        tokens_set.add(r)
    else:
        r = Response('Authorization rejected.', 401)
    return r


@app.route('/document', methods=['GET'])
@token_check
def get_doc():
    try:
        topic_id = request.args.get('topic_id')
        doc_name = request.args.get('doc_name')
        offset = [request.args.get('start', None),
                  request.args.get('end', None)]
        offset = [int(o) for o in offset if o]
        sentence = dm.get_doc(topic_id, doc_name, offset)
        r = jsonify({'sentence': sentence, 'offset': offset})
    except KeyError:
        r = Response('Requested resouce not found.', 404)
    return r


@app.route('/paragraph', methods=['GET'])
@token_check
def get_para():
    try:
        topic_id = request.args.get('topic_id')
        doc_name = request.args.get('doc_name')
        offset = [request.args.get('start'),
                  request.args.get('end')]
        offset = [int(o) for o in offset if o]
        paragraph = dm.get_para(topic_id, doc_name, offset)
        r = jsonify(dm.get_para(topic_id, doc_name, offset))
    except KeyError as e:
        print e
        r = Response('Requested resouce not found.', 404)
    return r


if __name__ == '__main__':
    app.run(host=opts.host, port=opts.port)
