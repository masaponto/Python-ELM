#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from elm.elm import ELM


class COBELM(ELM):
    """
    Equality Constrained-Optimization-Based ELM
    """

    def __init__(self, hid_num, a=1, c=2 ** 0):
        """
        Args:
        hid_num (int): number of hidden layer
        a (int) : const value of sigmoid funcion
        """
        super().__init__(hid_num, a)
        self.c = c

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] Array: feature vectors of learnig data
        y [float] Array: labels of leanig data
        """
        # number of class, number of output neuron
        self.out_num = max(y)
        x_vs = self._add_bias(X)

        # weight hid layer
        np.random.seed()
        self.a_vs = np.random.uniform(-1.0, 1.0, (len(x_vs[0]), self.hid_num))

        # output matrix hidden nodes
        h = self._sigmoid(np.dot(x_vs, self.a_vs))

        I = np.matrix(np.identity(len(h)))

        h_t = np.array(np.dot(h.T, np.linalg.inv(
            (I / self.c) + np.dot(h, h.T))))

        if self.out_num == 1:
            self.beta_v = np.dot(h_t, y)
        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            self.beta_v = np.dot(h_t, t_vs)


def main():

    db_name = 'iris'
    hid_nums = [10, 20, 30, 1000]
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    print('COBELM')
    for hid_num in hid_nums:
        print(str(hid_num), end=' ')

        e = COBELM(hid_num, c=2**5)
        ave = 0
        for i in range(10):
            scores = cross_validation.cross_val_score(
                e, data_set.data, data_set.target, cv=5, scoring='accuracy')
            ave += scores.mean()
        ave /= 10
        print("Accuracy: %0.2f " % (ave))


    print('ELM')
    for hid_num in hid_nums:
        print(str(hid_num), end=' ')
        e = ELM(hid_num)
        ave = 0

        for i in range(10):
            scores = cross_validation.cross_val_score(
                e, data_set.data, data_set.target, cv=5, scoring='accuracy')
            ave += scores.mean()
        ave /= 10
        print("Accuracy: %0.2f " % (ave))


if __name__ == "__main__":
    main()
