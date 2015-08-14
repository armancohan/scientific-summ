from __future__ import absolute_import
from __future__ import division, unicode_literals

import json
from argparse import ArgumentParser
from functions import annotations_functions as anns_func
import importlib
import numpy as np
import codecs
import os.path
import time
import constants
from collections import OrderedDict
try:
    import cPickle as pickle
except TypeError:
    import pickle
from libs.evaluate import (calculate_ndcg, calculate_ap,
                           format_results,
                           calculate_ar, set_folds, hide_reference_labels,
                           print_results, dump_stats, update_facet_count,
                           calculate_offset_stats, calc_rouge_scores)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


from util.annotations_client import AnnotationsClient
from _collections import defaultdict
from util.rouge.PythonROUGE.rouge_wrapper import calc_rouge
from random import randint
from copy import deepcopy
import itertools
from log_conf import Logger
from summarizer.mmr_summarizer import MMR

from util.aritmatic_operations import mean_conf
from util.tokenization import WordTokenizer
from util.common import write_json_as_csv, hash_obj, hash_dict
import gzip

w_t = WordTokenizer(stem=False)

logger = Logger(__file__.split('/')[-1]).logger

path = constants.get_path()
result_outpath = 'tmp/tmpres/'

_ANNS_DIR = path['ann']
_ANNS_PATH = path['ann_json']
CACHE = path['cache']

valid_topics = ['all']
# doc_mod = DocumentsModel(_ANNS_DIR)

CACHE_FILE = constants.join_path(
    CACHE, 'umls.json')
if os.path.isfile(CACHE_FILE):
    try:
        with codecs.open(CACHE_FILE, 'rb', 'utf-8') as mf:
            cachefile = json.load(mf)
    except:
        cachefile = {}
else:
    cachefile = {}


def evaluate(opts, args):
    def tofloat(s): return '%.3f' % s
    # round float representation

    def toCI(s): return s, s + '_cb', s + '_ce'
    # gets a measure, adds low and high confidence intervals
    doc_mod = AnnotationsClient(host=opts.ac_host, port=opts.ac_port)

    logger.info('opts: [ %s ]' %
                ('; '.join(['%s: %s' % opt
                            for opt in vars(opts).iteritems()])))
    logger.info('args: [ %s ]' % (' '.join([str(a) for a in args])))

    with file(opts.anns_path, 'rb') as af:
        annotations = json.load(af)
    if not opts.run_comment:
        opts.run_comment = 'nocomment'

    if not opts.summbaseline:  
        if opts.extract_summary:
            summ_rouge_scores = defaultdict(lambda: defaultdict(dict))
            sum_lengths = [100, 250]
            # load required data for summarization
            with open('data/task1b_results1.json') as mf:
                facets = json.load(mf)
            with codecs.open('data/v1-2b.json', 'rb', 'utf-8') as mf:
                ref_summs = json.load(mf)
            summarizerclass = importlib.import_module(
                'summarizer.' + opts.summarizer)
            summarizermethod = summarizerclass.Summarizer(args, opts)

        methodclass = importlib.import_module(opts.method)
        if opts.rerank is not None:
            rerankerclass = importlib.import_module('rerank.' + opts.rerank)
        folds = set_folds(annotations, opts)  # list(tuple(list(dict)))
        # The size of the tuple is 2 (train_data, test_data)
        # inner dict keys: 'citation_text', 'reference_text', etc
        ares = {}
        folds_cnt = 1
        dump_stats_data = []
        facet_cnt = {}  # dict for counting the facets for each citation

        rouges = []
        detailed_res = {}
        log_name = opts.method + '_' + time.strftime('%d-%b--%H-%M--%S')
        indiv_stats = defaultdict(lambda: defaultdict(list))

        method = methodclass.Method(args, opts)
        if opts.rerank is not None:
            reranker = rerankerclass.Reranker(args, opts)
            method_properties = {
                'method_name': opts.method,
                'reranker': opts.rerank}
            method_properties.update(method.method_opts)
            method_properties.update(reranker.reranker_opts)
            method_id = hash_dict(method_properties)
        else:
            method_properties = {'method_name': opts.method}
            method_properties.update(method.method_opts)
            method_id = hash_dict(method_properties)
        method_cache_path = CACHE + '/method_cache/' + method_id

        method_results = {}
        if not os.path.exists(method_cache_path):
            for train_data, test_data in folds:
                logger.info('fold %s/%s.' % (folds_cnt, len(folds)))
                method.train(train_data)
                extracted_refs = method.run(hide_reference_labels(test_data))
                if opts.rerank is not None:
                    extracted_refs = reranker.rerank(extracted_refs)
                method_results[extracted_refs[0][0]['topic']] = {'extracted_refs': extracted_refs,
                                                                 'test_data': test_data}
                folds_cnt += 1
            with gzip.open(method_cache_path, 'wb') as mf:
                pickle.dump(method_results, mf)
        else:
            with gzip.open(method_cache_path, 'rb') as mf:
                method_results = pickle.load(mf)

            # list(list(dict)) inner list is always size of size 1,
            # outer list size of citances for each topic
            # inner_dict.keys() : ['topic', 'citing_article', '_type',
            # '_id', '_score', '_index', 'sentence', 'query',
            # 'offset', 'citance_number']

#             if opts.method == 'method.umls_expand':
#                 with codecs.open(CACHE_FILE, 'wb', 'utf-8') as mf:
#                     json.dump(cachefile, mf)
        for ky, obj in method_results.iteritems():
            if valid_topics == ['all'] or ky in valid_topics:
                extracted_refs = obj['extracted_refs']
                test_data = obj['test_data']
                facet_cnt = update_facet_count(facet_cnt, test_data)
                if not opts.ignore_rouge:
                    rougel = calc_rouge_scores(
                        extracted_refs, test_data, doc_mod)
                    rouges.append(rougel[0:2])
                else:
                    rougel = (0, 0, 0, {})
                    # rougel[0:2] contains average rouge r,p,f1 for each facet
                prec, indiv_prec = calculate_ap(extracted_refs, test_data)
                rec, indiv_recall = calculate_ar(extracted_refs, test_data)
                indiv_f1 = {key: (2 * indiv_prec[key] * indiv_recall[key]) /
                            (indiv_prec[key] + indiv_recall[
                                key]) if (indiv_prec[key] + indiv_recall[key]) > 0
                            else 0.0 for key in indiv_prec}

                f1 = {fcet_name: (((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0)
                      for ((fcet_name, p), (_, r))
                      in zip(prec.iteritems(), rec.iteritems())}  # f1_by_facet
                ndcg, indiv_ndcg = calculate_ndcg(extracted_refs, test_data)

                indiv_stats['prec'].update(indiv_prec)
                indiv_stats['recall'].update(indiv_recall)
                indiv_stats['f1'].update(indiv_f1)
                indiv_stats['ndcg'].update(indiv_ndcg)
                metrics = [('Precision', prec),
                           ('Recall', rec),
                           ('F-1', f1),
                           ('nDCG', ndcg)]
                if not opts.ignore_rouge:
                    metrics.extend([('Rouge-L-r', rougel[0]),
                                    ('Rouge-L-p', rougel[1]),
                                    ('Rouge-L-f', rougel[2])])
                for n, res in (metrics):
                    for facet, facet_data in res.iteritems():
                        ares.setdefault(n, {}).setdefault(
                            facet, []).append(facet_data)
    #             if opts.offset_stats:  # TODO: Check
    #                 r = calculate_offset_stats(extracted_refs, test_data)
    # #                 r1 = rougel[0:2]
    #                 dump_stats_data.append(r[1])
    #                 dump_json_data.append(r1)
    #                     detailed_res = dict(
    #                         detailed_res.items() + rougel[0:2].items())

                if opts.extract_summary:  # Use citance matches for summary
                    # from summarizer.facet_summarizer import FacetSummarizer
                    # summarizer = FacetSummarizer()
                    for l in sum_lengths:
                        summary = summarizermethod.summarize(
                            extracted_refs, facets, max_length=l)
                        for k, summ in summary.iteritems():
                            logger.debug('Summary length: %d' % w_t.count_words(
                                ' '.join(summ)))
                            summ_rouge_scores[l][k] = calc_rouge(
                                [' '.join(summ)],
                                [ref_summs[k.lower()].values()], k)
                        logger.debug('%s, %d: %.3f' % (k, l, np.mean([e['rouge_l_f_score']
                                                                      for _, e
                                                                      in summ_rouge_scores[
                            l].iteritems()])))
                        run_id = opts.method + '_' +\
                            opts.summarizer + '_' + method_id + \
                            '_' + opts.run_comment
                        with codecs.open('tmp/summaries/' + run_id + '-' + str(l), 'wb', 'utf-8') as mf:
                            mf.write(' '.join(summ))

#                 for t in extracted_refs:
#                     summaries[t[0]['topic']][t[0]['citance_number']] =\
#                         t[0]['sentence']

        if opts.extract_summary:
            #             with codecs.open('tmp/followup_summaries.json', 'wb', 'utf-8') as mf:
            #                 json.dump(summary, mf, indent=2)
            measures = [e for e in summ_rouge_scores[100][
                'D1418_TRAIN'].keys() if e[-3:] != '_cb' and e[-3:] != '_ce']
#             measures = {'rouge_l_f_score, rouge_l_recall', 'rouge_l_precision',
#                         'rouge_1_f_score, rouge_1_recall', 'rouge_1_precision',
#                         'rouge_2_f_score, rouge_2_recall', 'rouge_2_precision',
#                         'rouge_3_f_score, rouge_3_recall', 'rouge_3_precision',
#                         'rouge_w_1.2_f_score', 'rouge_w_1.2_recall', 'rouge_w_1.2_precision'}

            for l in sum_lengths:
                with codecs.open('tmp/summ-scores-%s-%d.cpickle' %
                                 ('summary_by_refs', l),
                                 'wb', 'utf-8') as mf:
                    pickle.dump(summ_rouge_scores[l], mf)
                fscore = [e['rouge_l_f_score']
                          for _, e in summ_rouge_scores[l].iteritems()]
                recall = [e['rouge_l_recall']
                          for _, e in summ_rouge_scores[l].iteritems()]
                precision = [e['rouge_l_precision']
                             for _, e in summ_rouge_scores[l].iteritems()]
                logger.info('Rouge-L f_score for length %d summaries: %.3f +- %.3f' %
                            (l, np.mean(fscore), np.std(fscore)))
                logger.info('Rouge-L recall for length %d summaries: %.3f +- %.3f' %
                            (l, np.mean(recall), np.std(recall)))
                logger.info('Rouge-L precision for length %d summaries: %.3f +- %.3f' %
                            (l, np.mean(precision), np.std(precision)))
                if not os.path.exists('tmp/summ-scores.csv'):
                    with codecs.open('tmp/summ-scores.csv', 'a') as csvfile:
                        row = []
                        for m in measures:
                            row.extend(toCI(m))
                        csvfile.write(','.join(row) + '\n')
                run_id = opts.method + '_' +\
                    opts.summarizer + '_' + method_id + '_' + opts.run_comment
                write_json_as_csv(summ_rouge_scores[l],
                                  result_outpath + '/%s-%d.csv' % (run_id, l))
                logger.debug('run id: %s-%d' % (run_id, l))

                if not os.path.exists('tmp/results.keys'):
                    results_keys = OrderedDict()
                else:
                    with open('tmp/results.keys') as mf:
                        results_keys = json.load(
                            mf, object_pairs_hook=OrderedDict)
                if method_id not in results_keys:
                    results_keys[method_id] = [method_properties]

                with open('tmp/results.keys', 'w') as mf:
                    json.dump(results_keys, mf, indent=2)

                with codecs.open('tmp/summ-scores.csv', 'a') as csvfile:
                    row = [time.strftime('%d-%b--%H-%M--%S')]
                    row.extend([method_id])
                    row.extend([opts.summarizer])
                    row.extend([str(l)])
                    for measure in measures:
                        row.extend(map(tofloat, mean_conf([e[measure]
                                                           for _, e in summ_rouge_scores[l].iteritems()])))
#                     row.extend(map(tofloat, mean_conf(fscore)))
#                     row.extend(map(tofloat, mean_conf(([e['rouge_l_recall']
#                                                         for _, e in summ_rouge_scores[l].iteritems()]))))
#                     row.extend(map(tofloat, mean_conf([e['rouge_l_precision']
# for _, e in summ_rouge_scores[l].iteritems()])))
                    csvfile.write(','.join(row) + '\n')

        ares = {n: {facet: np.average(facet_data)
                    for facet, facet_data in res.iteritems()}
                for n, res in ares.iteritems()}
        print_results(ares, facet_cnt, opts.detailed)
    #     print "Rouge_L: " + str(tuple(map(np.mean, zip(*rouges))))
        log = time.strftime('[%d %b %y - %H:%M:%S] ') + \
            format_results(ares, facet_cnt, opts.detailed)
        method = " ::: method: %s, rrnk: %s, args: %s " %\
            (opts.method, opts.rerank, str(args))
        if not os.path.isdir('tmp/results/runs'):
            os.makedirs('tmp/results/runs')
        if not os.path.isdir('tmp/results/details'):
            os.mkdir('tmp/results/details')
        with open('tmp/results_all.log', 'a') as mf:
            mf.write(log + method + '\n')
        log1 = time.strftime('%d%b--%H-%M-%S') + \
            "-method-%s-rrnk-%s-args--%s" %\
            (opts.method, opts.rerank, str(args).replace(',', ''))
        if opts.log_comment is not None:
            log_name = opts.method + '_' + time.strftime('%d-%b--%H-%M--%S') +\
                '--' + opts.log_comment
        else:
            log_name = opts.method + '_' + time.strftime('%d-%b--%H-%M--%S')
        log_value = (opts.method, opts.rerank, str(args))
        with open('tmp/results/results_key.log', 'a') as mf:
            mf.write(log_name + '::::' + str(log_value) + '\n')
        with open('tmp/results/details/' + log_name, 'wb') as mf:
            json.dump(detailed_res, mf, indent=2, sort_keys=True)
        with open('tmp/results/runs/' + log_name, 'wb') as mf:
            json.dump(ares, mf, indent=2, sort_keys=True)

        if opts.dump_stats is not None:
            dump_stats(dump_stats_data, opts.dump_stats, opts.index_name)

    else:  # SUMMARIZATION
        sent_counts = [100, 250]

        stemmer = Stemmer('english')
        with codecs.open('data/v1-2b.json', 'rb', 'utf-8') as mf:
            ref_summs = json.load(mf)

        if opts.summgt:
            logger.info('Summarizing using gold reference spans')
#             print '---------------------------'
#             print 'getting gt for summ '
#             print '---------------------------'
            # ground truth spans from 1a to form summary
            with codecs.open('data/summ_reference.json', 'rb', 'utf-8') as mf:
                data = json.load(mf)
            rouge_scores = {}
            for l in sent_counts:
                summaries = {}
                for k in data:
                    summ = []
                    options = deepcopy(data[k])
                    maxx = set(
                        list(itertools.chain.from_iterable(options.values())))
                    while (len(summ) < min(l, len(maxx))):
                        for _, v in options.iteritems():
                            if len(v) > 0:
                                if len(v) > 1:
                                    idx = randint(0, len(v) - 1)
                                else:
                                    idx = 0
                                rr = v[idx]
                                del v[idx]
                                if len(summ) < l and rr not in summ:
                                    summ.append(rr)
                    summaries[k] = ' '.join(summ)
                    rouge_scores[k] = calc_rouge(
                        [' '.join(summ)], [ref_summs[k.lower()].values()], k)
#                     print 'rouge_scores: %s' %str(rouge_scores[k])
                with codecs.open('tmp/sum-scores-%s-%d.cpickle' %
                                 ('task1gt', l),
                                 'wb', 'utf-8') as mf:
                    pickle.dump(rouge_scores, mf)
                write_json_as_csv(rouge_scores, result_outpath + '/scores_%s-%d.csv' %
                                  ('sumgt', l))
                with codecs.open('tmp/summaries/summary-%s-%d.txt' %
                                 ('task1gt', l),
                                 'wb', 'utf-8') as mf:
                    json.dump(summaries, mf)
        if opts.crandom:
            logger.info('Summarizing using random citations')
            from summarizer.CRandom import Summarizer
            rouge_scores = {}
            with codecs.open('data/summ_citations.json', 'rb', 'utf-8') as mf:
                data = json.load(mf)
            citations = []
            for l in sent_counts:
                summaries = {}
                for k, v in data.iteritems():
                    c = v.values()
                    citations = list(itertools.chain.from_iterable(c))
                    cl = Summarizer()
                    summaries[k] = ' '.join(
                        cl.summarize(citations, max_length=l))
                    rouge_scores[k] = (
                        calc_rouge([summaries[k]],
                                   [ref_summs[k.lower()].values()], k))
                with codecs.open('tmp/summ-scores-%s-%d.cpickle' %
                                 (cl.__class__.__name__, l),
                                 'wb', 'utf-8') as mf:
                    pickle.dump(rouge_scores, mf)
                write_json_as_csv(rouge_scores, result_outpath + '/scores_%s-%d.csv' %
                                  (cl.__class__.__name__, l))
                with codecs.open('tmp/summaries/summary-%s-%d.txt' %
                                 (cl.__class__.__name__, l),
                                 'wb', 'utf-8') as mf:
                    json.dump(summaries, mf)

        if opts.clexrank:
            logger.info('Summarizing using lex rank with citations')
            from summarizer import CLexRank as c_l
            rouge_scores = {}
            with codecs.open('data/summ_citations.json', 'rb', 'utf-8') as mf:
                data = json.load(mf)
            citations = []

            for l in sent_counts:
                summaries = {}
                for k, v in data.iteritems():
                    c = v.values()
                    citations = list(itertools.chain.from_iterable(c))
                    cl = c_l.CLexRank()
                    summaries[k] = '. '.join(
                        cl.summarize(citations, max_length=l))
                    rouge_scores[k] = (
                        calc_rouge([summaries[k]],
                                   [ref_summs[k.lower()].values()], k))
                with codecs.open('tmp/summ-scores-%s-%d.cpickle' %
                                 (cl.__class__.__name__, l),
                                 'wb', 'utf-8') as mf:
                    pickle.dump(rouge_scores, mf)
                write_json_as_csv(rouge_scores, result_outpath + '/scores_%s-%d.csv' %
                                  (cl.__class__.__name__, l))
                with codecs.open('tmp/summaries/summary-%s-%d.txt' %
                                 (cl.__class__.__name__, l),
                                 'wb', 'utf-8') as mf:
                    json.dump(summaries, mf)

        if opts.mmr:
            logger.info('Summarizing using MMR')
            rouge_scores = {}
            with codecs.open('data/summ_citations.json', 'rb', 'utf-8') as mf:
                data = json.load(mf)
            citations = []
            lmb_vals = [0.3, 0.5, 0.9]
            for l in sent_counts:
                for lmb in lmb_vals:
                    summaries = {}
                    for k, v in data.iteritems():
                        c = v.values()
                        citations = list(itertools.chain.from_iterable(c))
                        mmr = MMR(lmbda=lmb)
                        summaries[k] = ' '.join(
                            mmr.summarize(citations, max_length=l))
                        rouge_scores[k] = \
                            calc_rouge([summaries[k]],
                                       [ref_summs[k.lower()].values()], k)
                    with codecs.open('tmp/summ-scores-%s-%d-%.2f.cpickle' %
                                     (mmr.__class__.__name__, l, lmb),
                                     'wb', 'utf-8') as mf:
                        pickle.dump(rouge_scores, mf)
                    write_json_as_csv(rouge_scores, result_outpath + '/scores_%s_%.2f-%d.csv' %
                                      (mmr.__class__.__name__, lmb, l))
                    with codecs.open('tmp/summaries/summary-%s-%d-%.2f.txt' %
                                     (mmr.__class__.__name__, l, lmb),
                                     'wb', 'utf-8') as mf:
                        json.dump(summaries, mf)
        if opts.lexrank:
            from sumy.summarizers.lsa import LsaSummarizer as LSASummarizer
            from sumy.summarizers.lex_rank import LexRankSummarizer as LEXSummarizer
            from sumy.summarizers.text_rank import TextRankSummarizer as TEXSummarizer
            from sumy.summarizers.luhn import LuhnSummarizer as LUNSummarizer

            lsa_summarizer = LSASummarizer(stemmer)
            lex_summarizer = LEXSummarizer(stemmer)
            tex_summarizer = TEXSummarizer(stemmer)
            lun_summarizer = LUNSummarizer(stemmer)
            methods = [
                lsa_summarizer]

            for num_sentences in sent_counts:
                for method in methods:
                    logger.info(
                        'running method %s with %d words' % (method, num_sentences))
                    method.stop_words = get_stop_words('english')
                    summaries = {}
                    docs = {}
                    for topic, anntator in annotations.iteritems():
                        for _, ann in anntator.iteritems():
                            ref = doc_mod.get_doc(topic.lower(),
                                                  ann[0]['reference_article'])['sentence']
                            docs[topic.lower()] = ref
            #                 import pdb; pdb.set_trace()
                    rouge_scores = {}
                    for k, v in docs.iteritems():
                        parser = PlaintextParser.from_string(
                            v, Tokenizer('english'))
                        doc_sum = []
                        for sentence in method(
                                parser.document, num_sentences):
                            if w_t.count_words(doc_sum) +\
                                    w_t.count_words(unicode(sentence)) < num_sentences:
                                doc_sum.append(unicode(sentence))
#                         while w_t.count_words(doc_sum) > num_sentences-20:
#                             doc_sum.pop()
                        rouge_scores[k] = \
                            calc_rouge(
                                [' '.join(doc_sum)], [ref_summs[k].values()], k)
                    with codecs.open('tmp/summ-scores-%s-%d.cpickle' %
                                     (method.__class__.__name__,
                                      num_sentences),
                                     'wb', 'utf-8') as mf:
                        pickle.dump(rouge_scores, mf)
                    write_json_as_csv(rouge_scores, result_outpath + '/scores_%s-%d.csv' %
                                      (method.__class__.__name__, num_sentences))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-v', '--verbose', dest="verbose",
                    default=False, action='store_true')
    ap.add_argument('-s', '--server', dest="server", default="localhost")
    ap.add_argument('-c', '--cachedir', dest="cachedir", default=CACHE)
    ap.add_argument('-p', '--port', dest="port", default=9200)
    ap.add_argument('-i', '--index-name', dest="index_name", default="tac")
    ap.add_argument('-m', '--method', dest="method", default=None)
    ap.add_argument('--ap', dest="anns_path", default=_ANNS_PATH,
                    help='location of annotation file (in JSON format).')
    ap.add_argument('-f', '--folds', default='auto', dest='folds',
                    help=('number of folds; choose between "auto" (splits by '
                          'topic_id) or define the number of folds.'))
    ap.add_argument('-a', '--ann-method', dest='ann_mthd',
                    default='simple_union',
                    choices=[m for m in dir(anns_func) if m.find('union') > 0])
    ap.add_argument('--dump-stats', default=None,
                    help='dump per-topic stats.')
    ap.add_argument('--dump-json', default=None,
                    help='dump per-topic results.')
    ap.add_argument('--ignore-rouge', default=False, action='store_true')
    ap.add_argument('--summbaseline', default=False, action='store_true')
    ap.add_argument('--clexrank', default=False, action='store_true')
    ap.add_argument('--mmr', default=False, action='store_true')
    ap.add_argument('--summgt', default=False, action='store_true')
    ap.add_argument('--lexrank', default=False, action='store_true')
    ap.add_argument('--extract-summary', default=False, action='store_true',
                    help='If true, summaries will be extracted from results of citation-context extraction')
    ap.add_argument('--summarizer', dest="summarizer", default=None, help='the '
                    'summarizer that will be used for summarizing extracted references')
    ap.add_argument('--crandom', dest='crandom', action='store_true')
    ap.add_argument('--detailed', default=False, action='store_true')
    ap.add_argument('--offset-stats', default=False, action='store_true')
    ap.add_argument('--rerank', default=None)
    ap.add_argument('--ac-host', default='localhost',
                    help='annotations client host')
    ap.add_argument('--ac-port', default=3003, type=int,
                    help='annotations client port')
    ap.add_argument('--ad', dest="anns_dir", default=_ANNS_DIR,
                    help='location of annotation original documents.')
    ap.add_argument('--log-comment', dest="log_comment", default=None,
                    help='Message appended at the end of the log name')
    ap.add_argument('--run-comment', dest="run_comment", default=None,
                    help='Message appended at the end of the run name')
    opts, args = ap.parse_known_args()

    # test if folds is in the right format;
    # if so, cast it to int; else, raise exception
    if not opts.folds == 'auto':
        try:
            opts.folds = int(opts.folds)
            assert opts.folds > 1, True
        except (ValueError, AssertionError), e:
            e.args = [('"%s" is not an acceptable folds value. Please specify '
                       'an integer greater than zero or use "auto".'
                       '') % opts.folds]
            raise

    evaluate(opts, args)
