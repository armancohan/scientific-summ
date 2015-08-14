from random import random
from libs.supervised.supervised_interface import SupervisedInterface


class Supervised(SupervisedInterface):

    supervised_opts = {'foo': {'default': 'bar'}}

    def train(self, X_train, y_train):
        return None

    def run(self, X_test):
        outcome = [round(random()) for d in X_test]
        return outcome

