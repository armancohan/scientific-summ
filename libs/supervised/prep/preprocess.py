'''
Created on Jan 11, 2015

@author: rmn
'''
from annotations_server.documents_model import DocumentsModel
import json
import codecs
import constants
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters


def union(s):
    '''
    return a list which is the union of a list of offsets s
    e.g. [[1,10],[5,15]] -> [[1,15]]
    '''
    s.sort(key=lambda x: x[0])
    y = [s[0]]
    for x in s[1:]:
        if y[-1][1] < x[0]:
            y.append(x)
        elif y[-1][1] >= x[0]:
            y[-1][1] = max(x[1], y[-1][1])
    return y


def sent_tokenize(data, filter_threshold=None):
    '''
    Tokenizes a string into sentences and corresponding offsets

    Args:
        data(str): The document itself
        filter_threshold(int): if sentence length is
            less than this, it will be ignored

    Returns:
        tuple(list(str), list(list))): tokenized
            sentences and corresponding offsets
    '''
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(
        ['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'et', 'al', 'Fig', 'fig'])
    sent_detector = PunktSentenceTokenizer(punkt_param)
    sentences = sent_detector.tokenize(data)
    offsets = sent_detector.span_tokenize(data)
    return (sentences, offsets)


def get_data(docs_path=constants.get_path()['ann'],
             json_data_path=constants.get_path()['ann_json']):
    '''
    Populates the docs_new object which stores information
        about the topics
        format of the docs_new:
        [ <list>(dict) keys (topic_id): 'd1418_train', ...:
            {'d1418_train' : [ <list>(dict) keys (citance_number): u'1', u'2', ...
                {u'11': [ <list>(dict) keys (annotator_id): 'I', 'B',...
                    {u'I': <dict>, keys:'ref_art',
                                    'not_relevant',
                                    'cit_offset',
                                    'cit_art',
                                    'ref_offset',
                                    'cit_text',
                                    'ref_text' }
    Args:
        docs_path(str): Path to the training data directory
            e.g. data/TAC_2014_BiomedSumm_Training_Data

        json_data_path(str): Path to the json training file (v1-2a.json)

    Returns:
        dict with the above format
    '''
    doc_mod = DocumentsModel(docs_path)
    docs = doc_mod.get_all()
    with codecs.open(json_data_path, 'rb', 'utf-8') as mf:
        data = json.load(mf)
    docs_new = {}
#     print docs.keys()
#     print docs.values()[0].keys()

    for tid, annotations in data.iteritems():
        if tid not in docs_new:
            docs_new[tid] = {}
        for annotator_id, ann_list in annotations.iteritems():
            for ann in ann_list:
                cit = ann['citance_number']
                if cit not in docs_new[tid]:
                    docs_new[tid][cit] = {}
                docs_new[tid][cit][annotator_id] = {}
                if 'ref_offset' not in docs_new[tid][cit][annotator_id]:
                    docs_new[tid][cit][annotator_id]['ref_offset'] =\
                        ann['reference_offset']
                else:
                    docs_new[tid][cit][annotator_id]['ref_offset'] = union(
                        docs_new[tid][cit][annotator_id]['ref_offset'] + ann['reference_offset'])
                if 'cit_offset' not in docs_new[tid][cit][annotator_id]:
                    docs_new[tid][cit][annotator_id]['cit_offset'] =\
                        [ann['citation_offset']]
                else:
                    docs_new[tid][cit][annotator_id]['cit_offset'] = union(
                        docs_new[tid][cit][annotator_id]['cit_offset'] + [ann['citation_offset']])
                docs_new[tid][cit][annotator_id][
                    'ref_art'] = ann['reference_article']
                docs_new[tid][cit][annotator_id][
                    'cit_art'] = ann['citing_article']

    for tid in docs_new:
        for cit in docs_new[tid]:
            for ann in docs_new[tid][cit]:
                docs_new[tid][cit][ann]['ref_text'] =\
                    [(s, doc_mod.get_doc(tid,
                                         docs_new[tid][cit][ann][
                                             'ref_art'].lower(),
                                         interval=s)) for s in
                     docs_new[tid][cit][ann]['ref_offset']]
                cit_off = union(docs_new[tid][cit][ann]['cit_offset'])
                docs_new[tid][cit][ann]['cit_text'] =\
                    ' '.join([doc_mod.get_doc(tid, docs_new[tid][cit][ann][
                        'cit_art'].lower(), intrvl) for
                        intrvl in cit_off])

    return docs_new

    #                     [(s, doc_mod.get_doc(tid,
    #                                          docs_new[tid][cit][ann][
    #                                              'cit_art'].lower(),
    #                                          interval=s)) for s in
    #                      docs_new[tid][cit][ann]['cit_offset']]

    # doc_mod =
    # DocumentsModel('../data/TAC_2014_BiomedSumm_Training_Data')
data = get_data()
# print data.values()[0].values()[0].values()
