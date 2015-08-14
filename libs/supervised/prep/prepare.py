'''
Created on Feb 12, 2015

Prepares the data for supervised classification

@author: rmn
'''

from util.es_interface import ESInterface
import pickle
import codecs
from preprocess import get_data, union
from util.clean_markers import clean


class Prep(object):

    '''
    Prepares data for supervised learners
    '''

    def __init__(self, index):
        self.es_int = ESInterface(index_name=index)

    def prep(self,
             docs_path='../data/TAC_2014_BiomedSumm_Training_Data',
             json_data_path='../data/v1-2a.json'):
        '''
        Converts the raw data into a list of sentences for each
            Topic.

        Args:
            docs_path (str), json_data_path (str)

        Returns:
            dict
                kes: topic_ids - UPPER CASE: e.g. D1410_TRAIN
                value: (list of tuples) - a list of training tuples
                    for format see _prep_data

        '''
        data = get_data(docs_path, json_data_path)
        train_set = {}
        for tid in data:
            train_set[tid] = []
            # citation number
            for cit in data[tid]:
                offsets = []
                ref_art = ''
                for ann in data[tid][cit].values():
                    for off in ann['ref_offset']:
                        offsets.append(off)
                    query = ann['cit_text']
                    ref_art = ann['ref_art']
                # union of all annotators reference offsets
                offsets = union(offsets)
                doc_type = tid.lower() + '_' + ref_art.lower()[:-4]
                d = self._prep_data(clean(query), doc_type, offsets)
                train_set[tid].extend(
                    d)
        return train_set

    def _prep_data(self, query, doc_type, relevant_offsets, save_path=False):
        '''
        Prepares the training data for leaning to rank
        Fetches the document from elastic_search and 
            returns a x_train, y_train vector


        Args:
            query(str) The query that is used to retrieve relevant offsets
            doc_type(str) Name of the type on elasticsearch index
                e.g. 'd1409_train_sherr'
            relevant_offsets(list): list of offsets that are relevant

        Returns:
            list of tuples: a list of training data
            ('query', 'some text', bool (1 if relevant 0 otherwise))
        '''
        hits = self.es_int.find_all(doc_type=doc_type.replace(',', ''))
        x_train = []
        y_train = []
        queries = []
        for hit in hits:
            label = 0
            offset = eval(hit['_source']['offset'])
            for off in relevant_offsets:
                if self.get_overlap(offset, off) > 0:
                    label = 1
                    break
            x_train.append(hit['_source']['sentence'])
            y_train.append(label)
            queries.append(query)
#         if save_path:
#             with codecs.open(save_path, 'wb', 'utf-8') as mf:
#                 pickle.dump(zip(x_train, y_train), mf)
        return zip(queries, x_train, y_train)

    def get_overlap(self, a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

if __name__ == '__main__':
    p = Prep()
    train = p.prep()
    from pprint import pprint
    pprint(train.keys())
    pprint(train["d1409_train".upper()][0:3])
#     p.prep_data("d1409_train_sherr", [
#                 [200, 500], [1000, 1200]], save_path='../cache/data.pickle')
