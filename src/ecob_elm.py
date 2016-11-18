#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


from sklearn.base import BaseEstimator, ClassifierMixin
from elm import ELM


class ECOBELM(ELM, ClassifierMixin):
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

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        X = self._add_bias(X)

        # weight hid layer
        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))

        # output matrix hidden nodes
        H = self._sigmoid(np.dot(self.W, X.T))
        I = np.matrix(np.identity(H.shape[0]))
        _H = np.array(np.dot(H.T, np.linalg.inv(
            (I / self.c) + np.dot(H, H.T))))

        self.beta = np.dot(_H.T, y)

        return self


def main():
    from sklearn import preprocessing
    from sklearn.datasets import fetch_mldata
    from sklearn.model_selection import cross_val_score

    db_name = 'iris'
    hid_num = 1000
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    print(db_name)
    print('ECOBELM', hid_num)
    e = ECOBELM(hid_num, c=2**5)
    ave = 0
    for i in range(10):
        scores = cross_val_score(
            e, data_set.data, data_set.target, cv=5, scoring='accuracy')
        ave += scores.mean()
    ave /= 10
    print("Accuracy: %0.2f " % (ave))

    print('ELM', hid_num)
    e = ELM(hid_num)
    ave = 0
    for i in range(10):
        scores = cross_val_score(
            e, data_set.data, data_set.target, cv=5, scoring='accuracy')
        ave += scores.mean()
    ave /= 10
    print("Accuracy: %0.2f " % (ave))


if __name__ == "__main__":
    main()
