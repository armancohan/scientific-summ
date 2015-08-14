'''
Created on Sep 19, 2014

@author: rmn
'''
from rerank.null import Reranker as RerankInterface
import json
import codecs
import os
import sys
from copy import deepcopy
from libs.evaluate import merge_offsets

# root_proj_path = os.getcwd()
# while not('.git' in os.listdir(root_proj_path)):
#     root_proj_path = os.path.split(root_proj_path)[0]
# if not(root_proj_path in sys.path):
#     sys.path.append(root_proj_path)


class Reranker(RerankInterface):

    reranker_opts = {'cutoff': {'type': int, 'default': 4},
                     'lookup': {'type': int, 'default': 4},
                     'relaxation': {'type': int, 'default': 0}}

    def __init__(self, args=None, opts=None):
        """ Initialize reranker.
        """
        super(Reranker, self).__init__(args, opts)
#         self.es_int = ESInterface(host=self.opts.server,
#                                   port=self.opts.port,
#                                   index_name=self.opts.index)
#         self.qsdata = json.load(file(self.opts.mlefn))

    def rerank(self, results):
        '''
        format of the results:
        [ [{_type, _index, sentence,offset]
          [{...}],
          ...
          []
        ]
        '''
#         with codecs.open('tmp/results.json', 'wb', 'utf-8') as mf:
#             json.dump(results, mf, indent=2)
        out = deepcopy(results)
        topop = []
        topopandrevers = []
        for res in results:
            for i in range(len(res[0]['offset'])):
                for j in range(i + 1, min(self.opts.lookup +
                                          self.opts.cutoff,
                                          len(res[0]['offset']))):
                    # look at lower ranked results, if there are overlaps,
                    # merge
                    newspan = merge_offsets(res[0]['offset'][i],
                                            res[0]['offset'][j],
                                            relaxation=self.opts.relaxation)
                    if newspan is not None:
                        res[0]['offset'][i] = newspan
                        topop.append(j)
            res[0]['offset'] = [
                x for idx1, x in enumerate(res[0]['offset']) if idx1 not in topop]
            res[0]['sentence'] = [
                x for idx1, x in enumerate(res[0]['sentence'])
                if idx1 not in topop][0:self.opts.cutoff]
            res[0]['offset'] = res[0]['offset'][0:self.opts.cutoff]
        return results
#     def _process_documents(self, results):
#         pass
if __name__ == '__main__':
    import codecs
    import json
    with codecs.open('/home/rmn/Downloads/tmp.json', 'rb', 'utf-8') as mf:
        results = json.load(mf)
    rrnk = Reranker()
    newres = rrnk.rerank(results)
    with codecs.open('/home/rmn/Downloads/tmp1.json', 'wb', 'utf-8') as mf:
        json.dump(newres, mf, indent=2)
