from __future__ import print_function
import errno
import itertools
import json
import hashlib
import re
import os
import sys
from functools import wraps
from constants import get_path
from pprint import pformat

from nltk.tokenize.regexp import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import codecs
import math
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()
tokenizer = RegexpTokenizer('[^\w\-\']+', gaps=True)
STOPWORDS = get_path()['data'] + '/stopwords.txt'
with file(STOPWORDS) as f:
    stopwords = frozenset([l.strip().lower() for l in f])

from optparse import OptionParser

from time import time as now

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass
try:
    import statsmodels.api as sm
except ImportError:
    pass


def plt_checker(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        try:
            plt.__doc__
            return method(*args, **kwargs)
        except NameError:
            return False
    return wrapper


def flatten(l):
    ''' flatten a list of lists into a single list
        from http://stackoverflow.com/questions/11264684/'''

    return [val for subl in l for val in subl]


def hash_file(fn):
    hasher = hashlib.md5()
    with open(fn, 'rb') as f:
        buf = f.read(2 ** 20)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(2 ** 20)

    return hasher.hexdigest()


def hash_dict(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True)).hexdigest()


def hash_obj(obj):
    try:
        return hashlib.md5(json.dumps(obj)).hexdigest()
    except TypeError:
        pass

    if type(obj) is dict:
        outobj = {}
        for k, v in obj.iteritems():
            try:
                outobj[k] = json.dumps(v)
            except TypeError:
                pass
    elif type(obj) in (list, tuple, set):
        outobj = []
        for v in obj:
            try:
                outobj.append(json.dumps(v))
            except TypeError:
                pass
    else:
        print('[error] obj can not be hashed')
        sys.exit(1)

    return hashlib.md5(json.dumps(outobj)).hexdigest()


def mkdir_p(path):
    ''' from http://stackoverflow.com/questions/600268'''
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


@plt_checker
def plot_ecdf(sample, xlabel, label, outdir=""):
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")

    plt.ylim(0, 1.0)

    plt.rcParams.update({'font.size': 14})
    plt.savefig(os.path.join(outdir, "%s_ecdf.pdf" % label), dpi=300)
    plt.close()


@plt_checker
def plot_hist(sample, xlabel, label, outdir=""):
    plt.hist(sample)
    plt.xlabel(xlabel)
    plt.ylabel("Count")

    plt.rcParams.update({'font.size': 14})
    plt.savefig(os.path.join(outdir, "%s_hist.pdf" % label), dpi=300)
    plt.close()


def parse_args(args=None, parser=None,
               add_method='add_option', parse_method='parse_args'):
    if (args is None) or isinstance(args, OptionParser):
        pargs = sys.argv
    else:
        pargs = args
    if parser is None:
        parser = (args if isinstance(args, OptionParser)
                  else OptionParser())
    getattr(parser, add_method)('-v', '--verbose', dest="verbose",
                                default=False, action='store_true')
    getattr(parser, add_method)('-s', '--server', dest="server",
                                default="localhost")
    getattr(parser, add_method)('-p', '--port', dest="port",
                                default=9200)
    getattr(parser, add_method)('-i', '--index-name', dest="index",
                                default="pubmed")

    return getattr(parser, parse_method)(pargs[1:])


def timer(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        printer = kwargs.pop('timer_printer', print)
        comment = kwargs.pop('timer_comment', '')
        start = now()
        resp = method(*args, **kwargs)
        elapsed = now() - start
        if elapsed > 3600:
            timestr = ('{:02.0f}:{:02.0f}:{:05.2f}'
                       '').format(elapsed / 3600,
                                  (elapsed % 3600) / 60,
                                  elapsed % 60)
        elif elapsed > 60:
            timestr = ('{:02.0f}:{:05.2f}'
                       '').format((elapsed % 3600) / 60,
                                  elapsed % 60)
        else:
            timestr = ('{:.3f} s'.format(elapsed))
        printer('[timer] %s%s executed in %s' %
                (method.__name__,
                 (' (%s)' % comment if comment else ''),
                 timestr))
        return resp
    return wrapper


# from http://rosettacode.org/wiki/Power_set#Python
def list_powerset(lst, include_empty=True):
    # the power set of the empty set has one element, the empty set
    result = [[]]

    for x in lst:
        # for every additional element in our set
        # the power set consists of the subsets that don't
        # contain this element (just take the previous power set)
        # plus the subsets that do contain the element (use list
        # comprehension to add [x] onto everything in the
        # previous power set)
        result.extend([subset + [x] for subset in result])

    if not include_empty:
        result.remove([])

    return result


def contains_sublist(lst, sublst):
    ''' from http://stackoverflow.com/questions/3313590/'''
    n = len(sublst)
    return any((sublst == lst[i:i + n]) for i in xrange(len(lst) - n + 1))


def prep_for_json(toprep):
    out = ''
    if type(toprep) in (list, tuple, set, frozenset):
        out = [prep_for_json(e) for e in toprep]
    elif type(toprep) is dict:
        out = {k: prep_for_json(v)
               for k, v in toprep.iteritems()}
    else:
        try:
            out = json.dumps(toprep)
        except TypeError:
            pass

    return out


def normalize_dictlist(obj_list, fields,
                       new_field=False, new_field_names=None, sum_to_1=False):
    '''Gets list of dictionaries (obj_list) and normalizes the values
     in obj[fields[i]], returns a dictionary with the
    same field normalized if
    new_field is false, if new_field is True, then adds
    a new field to the dicitonary with name new_field_name
    that contains normalized data'''
    for idx, fl in enumerate(fields):
        min_val = min(obj_list, key=lambda x: x[fl])[fl]
        max_val = max(obj_list, key=lambda x: x[fl])[fl]
        if sum_to_1:
            sum_val = sum(item[fl] for item in obj_list)
        new_list = []
        for obj in obj_list:
            if not new_field:
                if sum_to_1:
                    obj[fl] = obj[fl] / float(sum_val)
                else:
                    obj[fl] = (obj[fl] - min_val) / float(
                        max_val - min_val)

            else:
                if sum_to_1:
                    obj[new_field_names[idx]] = obj[fl] / float(sum_val)
                else:
                    obj[new_field_names[idx]] = (obj[fl] - min_val) / float(
                        max_val - min_val)

            new_list.append(obj)
    return new_list


class VerbosePrinter(object):

    def __init__(self, enabled=False, prefix=None):
        if prefix:
            self.prefix = '[{0}] '.format(prefix)
        else:
            self.prefix = False

        self.enabled = enabled

    def __call__(self, message, sep=' ',
                 end='\n', file_=sys.stdout):
        if self.enabled:
            if self.prefix:
                print('{p}{msg}'.format(p=self.prefix,
                                        msg=message),
                      sep=sep, end=end, file=file_)
            else:
                print(message, sep=sep, end=end, file=file_)


def tokenize(doc, stem=False, no_stopwords=True, lemmatize=False):
    """
    tokenizes a string

    Args:
        stem(bool)
        stopwords(bool)
        lemmatize(bool): Does the lemmatization just for nouns

    Returns:
        list(str) 
    """
    all_digits = re.compile(r"^\d+$").search
    terms = []
    for w in tokenizer.tokenize(doc.lower()):
        if no_stopwords:
            if w not in stopwords and (not all_digits(w)):
                if lemmatize:
                    terms.append(lmtzr.lemmatize(w))
                elif stem:
                    terms.append(stemmer.stem(w))
                else:
                    terms.append(w)
        else:
            if lemmatize:
                terms.append(lmtzr.lemmatize(w))
            elif stem:
                terms.append(stemmer.stem(w))
            else:
                terms.append(w)
    return terms


def write_json_as_csv(data, path, mode='w'):
    '''
    Writes a list of json data as a csv file
    [{obj1}, {obj2}, ... ]
    Only supports one level dictionaries

    Args:
        data(dict)
        path(str)
        mode(str) -- can be 'a' (append) or 'w' (write)
    '''
    csv = ''
    if not os.path.exists(path) or mode == 'w':
        s = ['key']
        s.extend(map(str, data.values()[0].keys()))
        csv = ','.join(s) + '\n'
    for k, v in data.iteritems():
        s = [k]
        s.extend(map(str, v.values()))
        csv += ','.join(s) + '\n'
    with codecs.open(path, mode + 'b', 'utf-8') as mf:
        mf.write(csv)
