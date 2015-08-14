from argparse import (ArgumentParser, Namespace)
import re


class SupervisedInterface(object):
    """docstring for SupervisedInterface"""

    supervised_opts = {}

    def __init__(self, args, opts):
        if opts:
            self.opts = opts
        else:
            self.opts = Namespace()

        self.args = args
        self._parse_args()

    def _parse_args(self):
        """ Parse the Reranker-specific parameters (as specified by
            supervised_opts) and add them to self.opts. Note: if an
            option is already specified in self.opts, it is NOT
            overwritten by this method.

            Notes
            ----------
            This method is NOT meant to be overwritten.
        """
        # replaces every char in every element of supervised_opts
        # that is not a dash, letter or digit with a dash.
        # It also convert the option to lowercase.
        fix_opts_names = re.compile(r'[^A-Za-z0-9\-]').sub
        self.supervised_opts = {fix_opts_names('-', k).lower(): v
                                for k, v in self.supervised_opts.iteritems()}

        method_parser = ArgumentParser()

        # Giant hack to overcome the fact that you can't unpack
        # with the star operator a vocabulary that has 'type'
        # as key. However, since ArgumentParser.add_argument
        # requires a 'type' parameter to specify the type of
        # an argument, I use a separate dictionary to collect
        # all the types that are specified in supervised_opts
        # and add them later by poking inside method_parser.
        types_map = {}

        for opt, values in self.supervised_opts.iteritems():
            types_map[opt] = values.pop('type', None)
            method_parser.add_argument(('--%s' % opt), **values)

        # Poking inside method_parser for the reasons described
        # above. The try...except sequence is necessary for when
        # opts is the help option, which is not present in types_map.
        for opt in (vars(o) for o in vars(method_parser)['_actions']):
            try:
                dest = opt['dest'].replace('_', '-')
                opt['type'] = types_map[dest]
            except KeyError:
                pass

        mopts, margs = method_parser.parse_known_args(self.args)
        self.args = margs
        for opt, value in vars(mopts).iteritems():
            vars(self.opts).setdefault(opt, value)

    def train(self, X_train, y_train):
        # X_train is a list of sentencences. y_train[j] == 1
        # iff X_train[j] belongs to the class SupervisedInterface
        # is being trained for; y_train[j] == 0 otherwise.
        return None

    def run(self, X_test):
        # return a list 'prediction' where the prediction[j] == 1
        # iff X_test[j] belongs to the class; predictions[j] == 0
        # otherwise.
        raise NotImplementedError()
