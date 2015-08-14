import ast
from collections import defaultdict
from copy import deepcopy
import csv
from itertools import chain
from math import ceil
import numpy
import os
from random import shuffle

from annotations_server.documents_model import DocumentsModel
import cPickle as pickle
from functions import annotations_functions as anns_func
import numpy as np
from util.cache import simple_caching
from util.common import hash_obj
from util.es_interface import ESInterface
# from util.rouge.PythonROUGE import PythonROUGE as rg
# from util.rouge.PythonROUGE.PythonROUGE import PythonROUGE
from util.rouge.PythonROUGE.rouge_wrapper import calc_rouge
#### UTILITIES ####
ANNS_DIR = 'data/TAC_2014_BiomedSumm_Training_Data'


def _gap(x):
    return float(x[1] - x[0])


def print_results(ares, facet_cnt, detailed):
    model = '''\n\t=== Results ===\n%s\n\t===============\n'''
    out = ''
    for n, res in ares.iteritems():
        if detailed:
            out += '\t%s\n' % n
        else:
            out += '\t%s: ' % n
        for facet, facet_data in sorted(res.items(), key=lambda t: t[0]):
            if not detailed and facet != u'all':
                continue
            if detailed:
                out += '\t\t%s: ' % ' '.join([e.capitalize()
                                              for e in facet.split('_')])
            out += '%.4f\n' % facet_data
    out = out[:-1]

    if detailed:
        out += '\n\tFacet Stats\n'
        for facet, facet_data in facet_cnt.iteritems():
            out += '\t\t%s: %s\n' % (facet, facet_data)
        out = out[:-1]

    print model % out


def format_results(ares, facet_cnt, detailed):
    out = ''
    for n, res in ares.iteritems():
        out += '%s: %f.3' % (n, res['all']) + "; "
    return out[:-2]


def dump_stats_json(dump_stats_data, dump_path):
    out_data = {}
#     for ann, res in dump_stats_data:
#         ann_id = '_'.join((ann['topic_id'][:-6].lower(),
#                            ann['citing_article'][:-4].lower(),
#                            ann['reference_article'][:-4].lower()))
#         out_data[ann_id] = {'citantion_text': ann['citation_text'],
#                             'gt_offset': str(ann['reference_offset']),
#                             'gt_text': ann['reference_text'],
#                             'citation_text_query': res[0]['query'],
#                             'sys_results': {
#                                 'offsets': [str(s['offset']) for s in res[:10]],
#                                 'sentences': [s['sentence'] for s in res[:10]]}
#                             }
    import codecs
    import json

    def convert_keys_to_string(dictionary):
        """Recursively converts dictionary keys to strings."""
        if not isinstance(dictionary, dict) and\
                (not isinstance(dictionary, list)) and\
                (not isinstance(dictionary, tuple)):
            return dictionary
        elif isinstance(dictionary, list):
            return [convert_keys_to_string(e) for e in dictionary]
        elif isinstance(dictionary, tuple):
            return str(dictionary)
        return dict((str(k), convert_keys_to_string(v))
                    for k, v in dictionary.items())
    tmp = []
    for k in dump_stats_data:
        tmp.append(convert_keys_to_string(k))
    with codecs.open(dump_path, 'w', 'UTF-8') as mf:
        json.dump(tmp, mf, indent=2, sort_keys=True)
        print 'results dumped in: ' + dump_path


def dump_stats(dump_stats_data, dump_path, index_name):

    es_int = ESInterface(index_name=index_name)
    csv_line = []

    for ann, res in dump_stats_data:
        csv_line.extend([[ann['topic_id'],
                          ann['citing_article'][:-4].lower(),
                          ann['reference_article'][:-4].lower(),
                          ann['discourse_facet']], [''],
                         [ann['citation_text'].encode('ascii', 'ignore')],
                         ['']])
        offsets = chain(*[[s[0], s[1], '']
                          for s in sorted(ann['reference_offset'].keys(),
                                          key=lambda t: t[0])])
        csv_line.extend([list(offsets), ['']])
        csv_line.append(['prec:'])
        csv_line.extend([list(t)
                         for t in calculate_ap([res], [ann]).items()])
        csv_line.append(['ndcg:'])
        csv_line.extend([list(t)
                         for t in calculate_ndcg([res], [ann]).items()])
        csv_line.append([''])
        for i, r in enumerate(res, start=1):
            rel = str(calculate_ndcg([[r]], [ann])['all'] > 0).upper()

            # temp until Arman fixes bug
            txt = es_int.get_page_by_res(
                r)['sentence'].encode('ascii', 'ignore')
            offset = str(es_int.get_page_by_res(r)['offset']).strip(
                '()').split(', ')
            csv_line.extend([[txt], ['rank', i, '', 'offset', offset[0],
                                     offset[1], '', 'rel?', rel]])

            # commented until bugs fixed
#             txt = []
#             for offset in r['offset']:
#                 txt.append(ann_cl.get_doc('_'.join(r['_type'].split('_')[:2]),
#                                           r['_type'].split('_')[2], offset))
#             txt = ' ... '.join(txt)
#             csv_line.extend([[txt], ['rank', i, '', 'offset',
#                                      r['offset'][0][0], r['offset'][0][1],
#                                      '', 'rel?', rel]])
#             csv_line.append([''])
        csv_line.extend([[''], ['']])

    with file(dump_path, 'wb') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(csv_line)

#### FUCTIONS ####


def _calculate_max_score(gt_offsets):
    score = np.sum([_gap(k) * v for k, v in gt_offsets.iteritems()])
    offset_length = np.sum([_gap(k) for k in gt_offsets])
    return score / offset_length


def _calculate_scores(offsets, gt):
    # weighted (by relevance) size of ground truth
    gt_size = np.sum([_gap(k) * v for k, v in gt.iteritems()])

    # weighted (by relevance) size of extracted annotation
    ann_size = np.sum([_gap(k) for k in offsets])

    # overlapping offset, with weights
    off_comp = _compare_all_offsets(offsets, gt)

    # portion of the ground thruth overlapping with the extracted offset
    # weighted by ground truth weights (those derived by overlap annotations)
    shared = sum([_gap(k) * v for k, v in off_comp['shared']])

    # size of the shared w/o weights
    total_gap = sum([_gap(k) for k, v in off_comp['shared']])

    score = {'_ndcg': ((shared / total_gap) if total_gap > 0 else 0.0),
             '_precision': shared / ann_size,
             '_recall': shared / gt_size}
    return score


def _compare_all_offsets(offsets, gt):
    offsets_dec = ([(p[0], 'start', 'ann', 1.0) for p in offsets] +
                   [(p[1], 'stop', 'ann', 1.0) for p in offsets])

    # CHEAP 'N' DIRTY FIXES
    # under some undetermined conditions gt is a
    # string rather than a dict. if that is the
    # case when executing, this simple fix prevents AttributeError
    if type(gt) is str:
        gt = eval(gt)

    grtruth_dec = ([(p[0], 'start', 'gt', v) for p, v in gt.iteritems()] +
                   [(p[1], 'stop', 'gt', v) for p, v in gt.iteritems()])
    all_pos = sorted(chain(offsets_dec, grtruth_dec), key=lambda t: t[0])
    last_pos = None
    segments = {'gt': [], 'ann': [], 'shared': []}
    ingt, inann, sc_gt, sc_ann = False, False, None, None
    for pos in all_pos:
        # deal with first iteration
        if last_pos is None:
            last_pos = pos[0]
            sc_gt = (pos[2] == 'gt') and pos[3]
            sc_ann = (pos[2] == 'ann') and pos[3]
            ingt, inann = (pos[2] == 'gt'), (pos[2] == 'ann')
            continue

        if ingt and inann and last_pos < pos[0]:
            segments['shared'].append(((last_pos, pos[0]), sc_gt))
        if ingt and not(inann) and last_pos < pos[0]:
            segments['gt'].append(((last_pos, pos[0]), sc_gt))
        if not(ingt) and inann and last_pos < pos[0]:
            segments['ann'].append(((last_pos, pos[0]), sc_ann))

        last_pos = pos[0]
        if (pos[2] == 'gt'):
            ingt = not(ingt)
            sc_gt = pos[3]
        if (pos[2] == 'ann'):
            inann = not(inann)
            sc_ann = pos[3]
    return segments


def _compute_annotations_score(annotations, ann_method,
                               cachedir, cache_comment):
    cache_path = os.path.join(cachedir, 'ann_score_%s.pickle' % cache_comment)
    if os.path.exists(cache_path):
        with file(cache_path, 'rb') as pf:
            annotations = pickle.load(pf)
    else:
        for ann_id, ann in annotations.iteritems():
            # select annotations
            ann['reference_offset'] = ann_method(ann['reference_offset'])
        with file(cache_path, 'wb') as pf:
            print '[cache] generating %s' % cache_path
            pickle.dump(annotations, pf)
    return annotations


#### LIB COMPONENT ####

def update_facet_count(facet_cnt, test_data):
    facets = set([e['discourse_facet'] for e in test_data] + facet_cnt.keys())
    facet_cnt = {f: (facet_cnt.setdefault(f, 0) +
                     len([e for e in test_data if e['discourse_facet'] == f]))
                 for f in facets}
    return facet_cnt


def calc_rouge_scores(extracted_refs, test_data,
                      doc_mod):
    """
    Calculates rouge-L scores for the extracted_references against the 
        Ground truth (test_data)

    Args:
        extracted_refs
        test_data -- ground truth data
        doc_mod -- document model that contains the actual document texts

    Returns(tuple)
    -------
    (avg rouge recall, avg rouge precision, avg rouge f1, detailed rouge scores)
    """
    retobj = []
    rouge_scores = defaultdict(lambda: defaultdict(dict))
    processed = {}
    rouglr = defaultdict(list)
    rouglp = defaultdict(list)
    rouglf = defaultdict(list)
    for gt, refs in zip(test_data, extracted_refs):
        facet = gt['discourse_facet'].lower()
        if len(refs) == 0:
            continue
        for ref in refs:
            if type(ref['offset']) is str:
                ref['offset'] = eval(ref['offset'])

        doc = doc_mod.get_doc(gt['topic_id'].lower(), gt['reference_article'])

        guess_sum = ' '.join([doc['sentence'][s[0]:s[1]]
                              for s in refs[0]['offset']])
        ref_sum = [' '.join([doc['sentence'][s[0]:s[1]] for s in ss])
                   for ss in gt['reference_offset_by_annotator']]
        key = gt['topic_id'].lower() + '--' + gt['citance_number']
        if key in processed:
            print "CITANCE ALREADY PROCESSED"
            rouge = processed[key]
        else:
            rouge = calc_rouge([guess_sum], [ref_sum], key)
            processed[key] = rouge
        rouglr[facet].append(rouge['rouge_l_recall'])
        rouglp[facet].append(rouge['rouge_l_precision'])
        rouglf[facet].append(rouge['rouge_l_f_score'])
        rouglr['all'].append(rouge['rouge_l_recall'])
        rouglp['all'].append(rouge['rouge_l_precision'])
        rouglf['all'].append(rouge['rouge_l_f_score'])

        rouge_scores[key] = rouge

    rouge_mean_r = {f: np.average(rouglr[f]) for f in rouglr}
    rouge_mean_p = {f: np.average(rouglp[f]) for f in rouglp}
    rouge_mean_f = {f: np.average(rouglf[f]) for f in rouglf}
    return (rouge_mean_r, rouge_mean_p, rouge_mean_f, rouge_scores)


def _get_uncovered_spans(offsets, gt_spans):
    uncovered = []
    for gt in gt_spans:
        covered = False
        for off in offsets:
            if (gt[0] <= off[0] and gt[1] >= off[0]) or\
                    (gt[0] >= off[0] and gt[0] < off[1]):
                covered = True
                break
        if not covered:
            uncovered.append(gt)
    return uncovered


def calculate_ndcg(extracted_refs, test_data):
    '''
    Calculates overall and individual ndcg values

    Returns: tuple(float, dict)
        see calculate_ap description for more info
    '''
    sum_ndcg = {}
    facet_cnt = {}
    ndcg_all = {}
    for gt, refs in zip(test_data, extracted_refs):
        facet = gt['discourse_facet'].lower()
        if len(refs) == 0:
            continue
        list_sc = {}
        key = gt['topic_id'].lower() + '_' + gt['citance_number']

        for rank, list_refs in enumerate(refs, start=1):
            # overlapping offset, with weights
            off_comp = _compare_all_offsets(list_refs['offset'],
                                            gt['reference_offset'])

            # portion of the ground thruth overlapping with
            # the extracted offset weighted by ground truth
            # weights (those derived by overlap annotations)
            shared = sum([_gap(k) * v for k, v in off_comp['shared']])

            # size of the shared w/o weights
            total_gap = sum([_gap(k) for k, v in off_comp['shared']])

            v = ((shared / total_gap) if total_gap > 0 else 0.0)
            list_sc[rank] = v

        dcg = np.sum([(v if r == 1 else (v / np.log2(r)))
                      for r, v in list_sc.iteritems()])
        idcg = np.sum([(v if r == 1 else (v / np.log2(r)))
                       for r, v in enumerate(sorted(list_sc.itervalues(),
                                                    reverse=True), start=1)])
        if dcg == idcg:
            ratio = 1.0 if (dcg > 0) else 0.0
        else:
            ratio = dcg / idcg
        ndcg_all[key] = ratio
        for s in (u'all', facet):
            sum_ndcg[s] = sum_ndcg.get(s, 0.0) + ratio
            facet_cnt[s] = facet_cnt.get(s, 0) + 1

    avg_ndcg = {f: (sum_ndcg[f] / size_f)
                for f, size_f in facet_cnt.iteritems()}
    return avg_ndcg, ndcg_all


def calculate_ar(extracted_refs, test_data):
    '''
    Calculates Average recall and also individual recalls

    Returns:
        tuple(average_recall, dict of recalls)
    '''
    sum_rec = {}
    facet_cnt = {}

    recall_all = {}
    # A dictionary of recalls by each topic and citation number
    # A dictionary of recalls by each facet

    for gt, refs in zip(test_data, extracted_refs):
        key = gt['topic_id'].lower() + '_' + gt['citance_number']
        facet = gt['discourse_facet'].lower()
        if len(refs) == 0:
            continue

        # CHEAP 'N' DIRTY FIXES
        # under undetermined conditions gt['reference_offset'] is a
        # string rather than a list of tuples. if that is the
        # case when executing, this simple fix prevents AttributeError
        if type(gt['reference_offset']) is str:
            gt['reference_offset'] = eval(gt['reference_offset'])

        # weighted (by relevance) size of ground truth
        gt_size = np.sum([_gap(k) * v
                          for k, v in gt['reference_offset'].iteritems()])

        # overlapping offset, with weights
        off_comp = _compare_all_offsets(refs[0]['offset'],
                                        gt['reference_offset'])

        # portion of the ground thruth overlapping with
        # the extracted offset weighted by ground truth
        # weights (those derived by overlap annotations)
        shared = sum([_gap(k) * v for k, v in off_comp['shared']])

        v = shared / gt_size
        recall_all[key] = v
        for s in (u'all', facet):
            sum_rec[s] = sum_rec.get(s, 0.0) + v
            facet_cnt[s] = facet_cnt.get(s, 0) + 1

    avg_rec = {f: (sum_rec[f] / size_f)
               for f, size_f in facet_cnt.iteritems()}

    return avg_rec, recall_all


# def calculate_rouge(extracted_refs, test_data, doc_mod, combined=True):
#     """
#     calculate rouge, assumes combined version
#         i.e. only one result
#
#     Returns:
#         tuple(list, dict, dict)
#             list: A list of rouge recall, precision, f1
#             dict: dict of individual rouge scores by topic and citance num
#             dict: dict of individual rouge scores by facet
#     """
#     scores = []
#     scores_all = defaultdict(list)
#     scores_facet = defaultdict(list)
#     for gt, refs in zip(test_data, extracted_refs):
#         key = gt['topic_id'].lower() + '_' + gt['citance_number']
#         facet = gt['discourse_facet'].lower()
#         if len(refs) == 0:
#             continue
#
#         # CHEAP 'N' DIRTY FIXES
#         # under undetermined conditions gt['reference_offset'] is a
#         # string rather than a list of tuples. if that is the
#         # case when executing, this simple fix prevents AttributeError
#         if type(gt['reference_offset']) is str:
#             gt['reference_offset'] = eval(gt['reference_offset'])
#         doc = doc_mod.get_doc(gt['topic_id'], gt['reference_article'])
#         guess_list = [doc['sentence'][s[0]:s[1]] for s in refs[0]['offset']]
#         ref_list = [doc['sentence'][s[0]:s[1]] for s in gt['reference_offset']]
#         scores.append(PythonROUGE(guess_list, ref_list))
#         scores_all[key].append(scores)
#         scores_facet[key].append(scores)

#     return tuple(map(np.mean, zip(*scores))), scores_all, scores_facet
    #(mean rouge_recall, rouge_prec, rouge_f1)


def calculate_ap(extracted_refs, test_data):
    '''
    Calculates average precision and individual precisions

    Returns
        tuple(float, dict, dict)
            float: average precision
            dict: dict of individual precisions
                key: topicId_citanceNum
                val: precision
    '''
    sum_prec = {}
    facet_cnt = {}
    precision_all = {}
    for gt, refs in zip(test_data, extracted_refs):
        key = gt['topic_id'].lower() + '_' + gt['citance_number']
        facet = gt['discourse_facet'].lower()
        if len(refs) == 0:
            continue

        # CHEAP 'N' DIRTY FIXES
        # under undetermined conditions ref['offset'] is a
        # string rather than a list of tuples. if that is the
        # case when executing, this simple fix prevents IndexError
        for ref in refs:
            if type(ref['offset']) is str:
                ref['offset'] = eval(ref['offset'])

        # weighted (by relevance) size of extracted annotation
        ann_size = np.sum([_gap(k) for k in refs[0]['offset']])

        # overlapping offset, with weights
        off_comp = _compare_all_offsets(refs[0]['offset'],
                                        gt['reference_offset'])

        # portion of the ground thruth overlapping with
        # the extracted offset weighted by ground truth
        # weights (those derived by overlap annotations)
        shared = sum([_gap(k) * v for k, v in off_comp['shared']])

        v = shared / ann_size
        precision_all[key] = v
        for s in (u'all', facet):
            sum_prec[s] = sum_prec.get(s, 0.0) + v
            facet_cnt[s] = facet_cnt.get(s, 0) + 1

    avg_prec = {f: (sum_prec[f] / size_f)
                for f, size_f in facet_cnt.iteritems()}

    return avg_prec, precision_all


def _compare_offsets(offsets, gt):
    segs = []
    tp = 0
    tf = 0
    fp = 0
    fn = 0
    for off in offsets:
        for go in gt:
            if go[0] <= off[0] and go[1] >= off[1]:
                tp += _gap(off)
            elif go[0] > off[0] and go[1] < off[1]:
                tp += _gap(go)
                fp += go[0] - off[0] + off[1] - go[1]
            elif go[1] > off[0] and go[1] <= off[1]:
                tp += go[1] - off[0]
                fp += off[0] - go[0]
                fn += off[0] - go[0]
            elif go[0] > off[0] and go[0] < off[1]:
                tp += off[1] - go[0]
                fp += go[0] - off[0]
                fn += go[1] - off[1]
            else:
                print "no match"
                fp += _gap(off)
                fn += _gap(go)
    if tp + fp != 0:
        p = tp / float((tp + fp))
    else:
        p = 0
    if tp + fn != 0:
        r = tp / float((tp + fn))
    else:
        r = 0
    f1 = 2 * p * r / float(p + r) if p + r > 0 else 0

#     for off in offsets:
#         cl = -1
#         for go in gt:
#             if go[0] <= off[0] and go[1] >= off[1]:
#                 cl = 0
# full overlap
#                 seg = {'state': 'fo', 'p': 1.0, 'r': (
#                     off[1] - off[0]) / float(go[1] - go[0]), 'd': 0}
#                 segs.append(seg)
#                 break
#         if cl == -1:
#             for go in gt:
# off: (300,500) gt:(200,400)
#                 tp = 0
#                 fp = 0
#                 tn = 0
#                 fn = 0
#                 if go[1] > off[0] and go[1] <= off[1]:
#                     if off[0] > go[0]:
#                         cl = 0
# partial overlap
#                         tp = go[1] - off[0]
#                         seg = {
#                             'state': 'po+', 'p': tp / float(off[1] - off[0]),
#                             'r': tp / float(go[1] - go[0]), 'd': off[1] - go[1]}
#                         segs.append(seg)
#                         break
#                     else:
#                         cl = 0
# partial overlap (sink)
#                         tp = go[1] - go[0]
#                         seg = {
#                             'state': 'po0', 'p': tp / float(off[1] - off[0]),
#                             'r': tp / float(go[1] - go[0]),
#                             'd': min(go[0] - off[0], off[1] - go[1])}
#                         segs.append(seg)
#                         break
# off: (100,300) gt:(200,400)
#                 elif go[0] > off[0] and go[0] <= off[1]:
#                     if go[1] >= off[1]:
#                         cl = 0
#                         tp = off[1] - go[0]
#                         seg = {'state': 'po-', 'p': tp / float(off[1] - off[0]), 'r': tp / float(
#                             go[1] - go[0]), 'd': go[0] - off[0]}
# partial overlap
#                         segs.append(seg)
#                         break
#                     else:
#                         cl = 0
#                         tp = go[1] - go[0]
#                         seg = {
#                             'state': 'po0', 'p': tp / float(off[1] - off[0]),
#                             'r': tp / float(go[1] - go[0]),
#                             'd': min(go[0] - off[0], off[1] - go[1])}
# sink
#                         segs.append(seg)
#                         break
# off: (100,300) gt:(200,250)
#                 elif go[0] > off[0] and go[1] < off[1]:
# partial overlap
#                     cl = 0
# partial overlap (sink)
#                     tp = go[1] - go[0]
#                     seg = {
#                         'state': 'po0', 'p': tp / float(off[1] - off[0]),
#                         'r': tp / float(go[1] - go[0]),
#                         'd': 0}
#                     segs.append(seg)
#                     break
#         if cl == -1:
# no overlap
#             try:
#                 closest_after = min([g[0] - off[1]
#                                      for g in gt if (g[0] - off[1] > 0)])
#             except:
#                 closest_after = 1000000
#             try:
#                 closest_before = min([off[0] - g[1]
#                                       for g in gt if (off[0] - g[1] > 0)])
#             except:
#                 closest_before = 1000000
#             seg = {
#                 'state': 'no', 'p': 0,
#                 'r': 0,
#                 'd': min(closest_before, closest_after)}
#             segs.append(seg)
    return p, r, f1


def calculate_offset_stats(extracted_refs, test_data):
    offsets_stats = {'top': [],
                     'rank_top': [],
                     'average': [],
                     'partial': [],
                     'full': [],
                     'discounted': [],
                     'not_overlap_cnt': []}
    res = {}
    for gt, refs in zip(test_data, extracted_refs):
        if len(refs) == 0:
            continue
        offsets = []
        ann_id = '_'.join((gt['topic_id'][:-6].lower(),
                           gt['citance_number'].lower()))
        for i, r in enumerate(refs, start=1):
            # overlapping offset, with weights
            segs = _compare_offsets(r['offset'], gt['reference_offset'])
            off_comp = _compare_all_offsets(r['offset'],
                                            gt['reference_offset'])

            offsets.append((i, segs))
        offsets_stats['rank_top'].append(offsets[0][1])
        offsets_stats['partial'].append(
            [o['d'] for o in [o1[1] for o1 in offsets][0] if 'po' in o['state']])
        offsets_stats['full'].append(
            [o['d'] for o in [o1[1] for o1 in offsets][0] if 'fo' in o['state']])

        offsets_stats['not_overlap_cnt'].append(
            len([o['state'] for o in [o1[1] for o1 in offsets][0] if 'no' in o['state']]))
        offsets_stats['rank_top'].append(offsets[0][1][0])
#         s1 = [k[1] for k in [o[1] for o in [o1[1] for o1 in offrsets][0]]] if len(offsets) > 0 else 0.0
#         print s1
        offsets_stats['average'].append(
            np.average([o['d'] for o in [o1[1] for o1 in offsets][0]]) if len(offsets) > 0 else 0.0)
        ext_offsets, ext_sentences, ext_stats, query = [], [], [], []
        for r in refs:
            ext_offsets.append(r['offset'])
            ext_sentences.append(r['sentence'])
            ext_stats.append(segs)
            query.append(r['query'])
        res[ann_id] = ({'original_query': gt['citation_text'],
                        'sys_query': query,
                        'gt': gt['reference_offset'],
                        'gt_text': gt['reference_text'],
                        'sys_res_offsets': ext_offsets,
                        'sys_res_sentences': ext_sentences,
                        'sys_stats': ext_stats})
#         offsets_stats['discounted'].append(
# np.average([o[1][1] / o[0] for o in offsets]) if len(offsets) > 0 else
# 0.0)
#     avg_stats = {k: 0 for k in offsets_stats}
#     for v1 in [v for k, v in offsets_stats.iteritems() if k != 'rank_top']:
#         if (v1 == 0 or v1 == 5):
#             avg_stats[k] = np.average(v1)
    avg_stats = {k: np.average(v1) for idx, v1 in enumerate(
        [v for k, v in offsets_stats.iteritems() if k != 'rank_top']) if idx == 0 or idx == 5}
    for k in offsets_stats.keys():
        if k not in avg_stats:
            avg_stats[k] = 0
    return (avg_stats, res)


def calculate_offset_stats_orig(extracted_refs, test_data):
    offsets_stats = {'top': [],
                     'rank_top': [],
                     'average': [],
                     'discounted': [],
                     'not_overlap_cnt': []}
    for gt, refs in zip(test_data, extracted_refs):

        if len(refs) == 0:
            continue

        offsets = []
        not_overlap_cnt = 0
        for i, r in enumerate(refs, start=1):
            # overlapping offset, with weights
            off_comp = _compare_all_offsets(r['offset'],
                                            gt['reference_offset'])
            # print off_comp
            if len(off_comp['shared']) > 0:
                continue

            not_overlap_cnt += 1
            closest_before = min([min([float(ro[0] - go[1])
                                       for go in gt['reference_offset'].keys()
                                       if ro[0] > go[1]] + [float('inf')])
                                  for ro in r['offset']])
            closest_after = min([min([float(go[0] - ro[1])
                                      for go in gt['reference_offset'].keys()
                                      if go[0] > ro[1]] + [float('inf')])
                                 for ro in r['offset']])
            closest = min(closest_after, closest_before)
            offsets.append((i, closest))

        offsets_stats['top'].append(offsets[0][1] if len(offsets) > 0 else 0)
        offsets_stats['not_overlap_cnt'].append(not_overlap_cnt)
        offsets_stats['rank_top'].append(
            offsets[0][0] if len(offsets) > 0 else 0)
        offsets_stats['average'].append(
            np.average([o[1] for o in offsets]) if len(offsets) > 0 else 0.0)
        offsets_stats['discounted'].append(
            np.average([o[1] / o[0] for o in offsets]) if len(offsets) > 0 else 0.0)

    avg_stats = {k: np.average(v) for k, v in offsets_stats.iteritems()}
    return avg_stats


@simple_caching()
def _union_citance(data, opts):
    data_copy = deepcopy(data)
    dc = DocumentsModel(opts.anns_dir)
    for ann_id in data_copy:
        txt = dc.get_doc(data_copy[ann_id]['topic_id'],
                         data_copy[ann_id]['citing_article'],
                         data_copy[ann_id]['citation_offset'])
        data_copy[ann_id]['citation_text'] = txt
    return data_copy


def set_folds(annotations, opts):
    data = {}
    ann_mthd = getattr(anns_func, opts.ann_mthd)

    # if true annoations doesn't have citation text/offest
    # because the data has no annotation about them
    eval_phase = False

    for ann in chain(*[chain(*[ann for ann in topic.itervalues()])
                       for topic in annotations.itervalues()]):

        # print ann['citation_offset'], type(ann['citation_offset'])
        # import sys
        # sys.exit(1)

        # [topic id]_[citing article name]_[reference article name]
        ann_id = '_'.join((ann['topic_id'][:-6].lower(),
                           ann['citing_article'][:-4].lower(),
                           ann['reference_article'][:-4].lower(),
                           ann['citance_number']))

        data.setdefault(ann_id,
                        {'discourse_facet': ann.get('discourse_facet', None),
                         'annotation_id': ann_id,
                         'topic_id': ann['topic_id'],
                         'citing_article': ann['citing_article'],
                         'reference_article': ann['reference_article'],
                         'citation_offset': ann['citation_offset'],
                         'citance_number': ann['citance_number'],
                         'citation_marker_offset': ann['citation_marker_offset'],
                         'citation_marker': ann['citation_marker'],
                         'citation_text': "",
                         'reference_text': [],
                         'reference_offset': [],
                         'reference_offset_by_annotator': []})
        if ann['citation_offset'][0] < data[ann_id]['citation_offset'][0]:
            data[ann_id]['citation_offset'][0] = ann['citation_offset'][0]
        if ann['citation_offset'][1] > data[ann_id]['citation_offset'][1]:
            data[ann_id]['citation_offset'][1] = ann['citation_offset'][1]
        if ((ann['citation_offset'][1] <
                data[ann_id]['citation_offset'][0]) or
                (ann['citation_offset'][0] >
                 data[ann_id]['citation_offset'][1])):
            print('WARNING: citation_offset not in agreement')
        try:
            data[ann_id]['reference_offset'].append(ann['reference_offset'])
            data[ann_id]['reference_offset_by_annotator'].append(
                ann['reference_offset'])
            reftext = ann['reference_text'].split(' ... ')
            data[ann_id]['reference_text'].extend(reftext)
        except KeyError:
            eval_phase = True

    data = _union_citance(data, opts, cachedir=opts.cachedir,
                          cache_comment=hash_obj(data))
    if eval_phase or vars(opts).get('eval_phase', False):
        return data.values()

        data.setdefault(ann_id, {'discourse_facet': ann['discourse_facet'],
                                 'annotation_id': ann_id,
                                 'topic_id': ann['topic_id'],
                                 'citing_article': ann['citing_article'],
                                 'reference_article': ann['reference_article'],
                                 'citation_offset': ann['citation_offset'],
                                 'citance_number': ann['citance_number'],
                                 'citation_marker': ann['citation_marker'],
                                 'citation_marker_offset': ann['citation_marker_offset'],
                                 'citation_text': ann['citation_text'],
                                 'reference_text': [],
                                 'reference_offset': []})
        data[ann_id]['reference_offset'].append(ann['reference_offset'])
        s = ' ... '
        data[ann_id]['reference_text'].extend(ann['reference_text'].split(s))
    data = _compute_annotations_score(data, ann_mthd, opts.cachedir,
                                      hash_obj((data, opts.ann_mthd)))

    if opts.folds == 'auto':
        groups = {}
        for ann in data.itervalues():
            groups.setdefault(ann['topic_id'], [])
            groups[ann['topic_id']].append(ann)
        groups = groups.values()
    else:
        ungrouped = data.values()
        shuffle(ungrouped)
        size_nth = int(ceil(len(ungrouped) / float(opts.folds)))
        groups = [ungrouped[(size_nth * i):(size_nth * (i + 1))]
                  for i in range(opts.folds)]

    folds = [(list(chain(*(groups[:i] + groups[(i + 1):]))), groups[i])
             for i in range(len(groups))]

    return folds


def hide_reference_labels(test_fold):
    out = []
    for ann in test_fold:
        oann = deepcopy(ann)
        oann.pop('reference_offset')
        oann.pop('reference_text')
        out.append(oann)
    return out


def merge_offsets(s, t, mode='U', relaxation=None):
    '''
    merges two offsets if they have overlapping spans,
    and returns the uniun of the two
    returns null if the intersection is null
    valid modes = U for Union, I for intersection
    e.g. merge_offsetes([5,10],[8,12])=[5,12]
    relaxation allows gaps between two spans
    '''
    if relaxation is None:
        if (s[1] < t[0]) or\
                (s[0] > t[1]):
            return None
        elif (s[0] <= t[0]):
            if s[1] <= t[1]:
                return [s[0], t[1]]
            else:
                return [s[0], s[1]]
        elif (s[0] >= t[0]) and (s[1] > t[1]):
            return [t[0], s[1]]
        elif (s[0] >= t[0]) and (s[1] <= t[1]):
            return t
    else:
        if (s[1] < t[0] - relaxation) or\
                (s[0] > t[1] + relaxation):
            return None
        elif (s[0] <= t[0] - relaxation):
            if s[1] <= t[1] + relaxation:
                return [s[0], t[1]]
            else:
                return [s[0], s[1]]
        elif (s[0] >= t[0] - relaxation) and (s[1] > t[1] - relaxation):
            return [t[0], s[1]]
        elif (s[0] >= t[0] - relaxation) and (s[1] <= t[1] + relaxation):
            return t

# print _compare_offsets([(0, 10), (15, 20), (50, 60)], [(5, 10), (15, 20)])
# gt = {(16918, 17516): 1.0, (13188, 13444): 1.0, (13041, 13186): 1.0, (19957,
#                                                                       21092): 1.0, (26148, 26322): 1.0, (12298, 12477): 1.0, (30079, 30206): 1.0}
# ref = [(33339, 33498), (18876, 19085), (32947, 33165), (20942, 21093), (32667, 32842), (31927, 32194), (17331, 17517), (1561, 1824), (32947, 33338), (18776, 19085), (32843, 33165), (19699, 19955), (33166, 33498), (31794, 32194), (21480, 21994),
#        (16088, 16319), (32667, 32946), (21658, 21994), (31927, 32481), (33339, 33672), (17130, 17517), (32482, 32842), (16088, 16424), (19699, 20047), (18876, 19317), (19479, 19955), (17331, 17827), (21658, 22379), (20942, 21479), (21095, 21657)]

# import codecs, json
# with codecs.open('../data/v1-2a.json', 'rb', 'utf-8') as mf:
#     data = json.load(mf)
# opts = {'ann_mthd':'test'}
# set_folds(data, opts)
