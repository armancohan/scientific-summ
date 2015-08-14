import os
import sys
from uuid import uuid4

# BEGIN PYTHONPATH FIX #
# appends root of project if PYTHONPATH is not set;
# relies on the presence of .git in project root.
root_proj_path = os.getcwd()
while not('.git' in os.listdir(root_proj_path)):
    root_proj_path = os.path.split(root_proj_path)[0]
if not(root_proj_path in sys.path):
    sys.path.append(root_proj_path)
# END  PYTHONPATH  FIX #

from method import (metamap,
                    condprob,
                    method_interface,
                    simple)
from util.es_interface import ESInterface


class InsertionRankMethodInterface(method_interface.MethodInterface):

    """ Wrapper around MethodInterface base class
        for query expansion.
        Uses a dictionary instead of reading the
        document/queries to expand from file.
    """

    def __init__(self,
                 documents,
                 eshost='localhost',
                 esport=9200,
                 esindex='pubmed21',
                 cachedir='cache'):

        self.cachedir = cachedir
        self.questions = documents
        self.categories = None

        self.added = dict([(qid, []) for qid in self.questions.keys()])
        self.removed = dict([(qid, []) for qid in self.questions.keys()])

        self.es = ESInterface(host=eshost, port=esport, index_name=esindex)

        self.tokenquestions = self.tokenize_questions(self.questions.items())
        self.tokquestions = dict([(k, " ".join(v)) for k, v in
                                  self.tokenquestions.iteritems()])

        self.run()

    def write_queries(self, queries):
        self.objout = queries


# class MedSyn(InsertionRankMethodInterface, medsyn.Method):
#
#     """MedSyn wrapper"""
#
#     def __init__(self, documents):
#         InsertionRankMethodInterface.__init__(self, documents)
#
#     def run(self):
#         medsyn.Method.run(self, [])


class MetamapExpand(InsertionRankMethodInterface, metamap.Method):

    """MetamapExpand wrapper"""

    def __init__(self, documents, tty_group=['syns', 'drugs']):
        self.args = [None, None, None]
        self.args.extend(tty_group)
#         metamap.Method.check_args(self, self.args)
        InsertionRankMethodInterface.__init__(self, documents)

    def run(self):
        metamap.Method.run(self, self.args)


# class MetamapSelect(InsertionRankMethodInterface, metamapselect.Method):
#
#     """MetamapSelect wrapper"""
#
#     def __init__(self, documents):
#         InsertionRankMethodInterface.__init__(self, documents)
#
#     def run(self):
#         metamapselect.Method.run(self, [])


class CondProb(InsertionRankMethodInterface, condprob.Method):

    """CondProb wrapper"""

    def __init__(self, documents, odds=0.5, _type='wiki'):
        self.args = [None, None, None]
        self.args.extend([odds, _type])
        condprob.Method.check_args(self, self.args)
        InsertionRankMethodInterface.__init__(self, documents)

    def run(self):
        condprob.Method.run(self, self.args)


class Simple(InsertionRankMethodInterface, simple.Method):

    def __init__(self, documents):
        InsertionRankMethodInterface.__init__(self, documents)

    def run(self):
        simple.Method.run(self, [])

if __name__ == '__main__':
    # tests if everything is working as it should
    # uses random string to ensure that cache is not
    # used!

    random_string = str(uuid4())
    test = {random_string: (u'I need some paracetamol; oh, and xanax!. ' +
                            'My bowel hurts and I have a sore foot.')}

    # print MedSyn(test).objout.items()[0][1]
    print CondProb(test).objout.items()[0][1]
    # print MetamapExpand(test).objout.items()[0][1]
    # print MetamapSelect(test).objout.items()[0][1]
