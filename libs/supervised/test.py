import codecs
import json
import os
from itertools import chain
from random import shuffle
from argparse import ArgumentParser
from math import ceil
from importlib import import_module
from util.common import VerbosePrinter

import numpy as np
from sklearn import metrics

METRICS = ('precision', 'recall', 'f1')

ap = ArgumentParser()
ap.add_argument('--ann-path', default='data/v1-1a.json')
ap.add_argument('--folds', default=4, type=int)
ap.add_argument('--supervised', default='randomized',
                choices=[f[:-3]
                         for f in os.listdir('libs/supervised/classifiers')
                         if f[-2:] == 'py' and f.find('__init__') < 0])
ap.add_argument('--rigen', default=None, action='store_true')
ap.add_argument('--cachedir', default='cache')
ap.add_argument('--tempdir', default='tmp')
ap.add_argument('-v', '--verbose', dest='verbose', default=False,
                action='store_true')
ap.add_argument('--detailed', default=False,
                action='store_true')
opts, args = ap.parse_known_args()

printer = VerbosePrinter(opts.verbose)

printer('opts: [ %s ]' % ('; '.join(['%s: %s' % opt
                                     for opt in vars(opts).iteritems()])))
printer('args: [ %s ]' % (' '.join([str(a) for a in args])))

if opts.rigen:
    for f in (e for e in os.listdir('cache') if e.find('clean') >= 0):
        os.remove(os.path.join('cache', f))

with codecs.open(opts.ann_path, encoding='utf-8') as af:
    annotations = json.load(af)

lines = []

rr = lambda x: '-'.join([str(e) for e in x])

facets_annotations = {}
overall = {m: [] for m in METRICS}

for topics in annotations.itervalues():
    for annotators in topics.itervalues():
        for ann in annotators:
            facet = ann['discourse_facet']
            reftexts = ann['reference_text'].split(' ... ')
            facets_annotations.setdefault(facet, []).extend(reftexts)

for facet in facets_annotations:
    results = {m: [] for m in METRICS}
    data = ([(e, 1) for e in facets_annotations[facet]] +
            [(e, 0) for e in chain(*[ann for f, ann in
                                     facets_annotations.iteritems()
                                     if f != facet])])
    shuffle(data)
    fsz = int(ceil(len(data) / float(opts.folds)))
    data = [data[i * fsz:(i + 1) * fsz] for i in range(opts.folds)]
    folds = [(list(chain(*(data[:i] + data[i + 1:]))), data[i])
             for i in range(len(data))]

    sup_class = import_module('libs.supervised.classifiers.%s' %
                              opts.supervised).Supervised

    for train_data, test_data in folds:
        X_train = np.array([e[0] for e in train_data])
        y_train = np.array([e[1] for e in train_data])
        X_test = np.array([e[0] for e in test_data])
        y_test = np.array([e[1] for e in test_data])

        sup = sup_class(args, opts)
        sup.train(X_train, y_train)
        predicted = sup.run(X_test)

        for m in METRICS:
            results[m].append(getattr(metrics,
                                      '%s_score' % m)(y_test, predicted))
            overall[m].append(getattr(metrics,
                                      '%s_score' % m)(y_test, predicted))
    results = {m: np.average(results[m]) for m in results}
    if opts.detailed or opts.verbose:
        print "%s (%s elements)" % (facet.replace('_', ' '),
                                    len(facets_annotations[facet]))
        for m in METRICS:
            print '\t%s: %.4f' % (m.capitalize(), results[m])

if opts.detailed or opts.verbose:
    print('==============')
overall = {m: np.average(results[m]) for m in overall}
print ('Overall (%s elements)' %
       sum([len(f) for f in facets_annotations.itervalues()]))
for m in METRICS:
    print '\t%s: %.4f' % (m.capitalize(), overall[m])
