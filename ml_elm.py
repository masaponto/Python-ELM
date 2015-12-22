#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from elm import ELM


class MLELM(ELM):

    def __init__(self,
                 hidden_units=[10, 20, 30]):
        self.hidden_units = hidden_units
        self.betas = []
        self.a = 1

    def fix(self, x_vs):
        for beta in self.betas:
            x_vs = self._add_bias(np.dot(x_vs, beta.T))
        return x_vs

    def fit(self, X, y):

        X = self._add_bias(X)
        self.out_num = max(y)

        for hid_num in self.hidden_units[:len(self.hidden_units) - 1]:

            x_vs = self.fix(X)

            np.random.seed()
            a_vs = np.random.uniform(-1., 1.,
                                     (len(x_vs[0]), hid_num))

            h_t = np.linalg.pinv(self._sigmoid(np.dot(x_vs, a_vs)))
            beta = np.dot(h_t, x_vs)

            self.betas.append(beta)

        hid_num = self.hidden_units[-1]
        x_vs = self.fix(X)
        np.random.seed()
        self.a_vs = np.random.uniform(-1., 1.,
                                      (len(x_vs[0]), hid_num))

        h_t = np.linalg.pinv(self._sigmoid(np.dot(x_vs, self.a_vs)))

        # find weights between output layer
        if self.out_num == 1:
            self.beta_v = np.dot(h_t, y)
        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            self.beta_v = np.dot(h_t, t_vs)

    def predict(self, X):
        x_v = self.fix(self._add_bias(X))
        g = self._sigmoid(np.dot(x_v, self.a_vs))
        return np.array(list(map(self._vtol, np.sign(np.dot(g, self.beta_v)))))


def main():
    #db_name = 'diabetes'
    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    print(data_set.data.shape)

    #e = ELM(50)
    #e.fit(data_set.data, data_set.target)
    #re = e.predict(data_set.data)
    #print(sum([r == y for r, y in zip(re, data_set.target)]) /
    #      len(data_set.target))

    e = ELM(50)
    ave = 0
    for i in range(10):
        scores = cross_validation.cross_val_score(
            e, data_set.data, data_set.target, cv=5, scoring='accuracy')
        ave += scores.mean()
    ave /= 10
    print("ELM Accuracy: %0.3f " % (ave))



    #e = MLELM(hidden_units=(10, 10, 50))
    #e.fit(data_set.data, data_set.target)
    #re = e.predict(data_set.data)
    #print(sum([r == y for r, y in zip(re, data_set.target)]) /
    #      len(data_set.target))


    e = MLELM(hidden_units=(10, 10, 50))
    ave = 0
    for i in range(10):
        scores = cross_validation.cross_val_score(
            e, data_set.data, data_set.target, cv=5, scoring='accuracy')
        ave += scores.mean()
    ave /= 10
    print("ML ELM Accuracy: %0.3f " % (ave))



if __name__ == "__main__":
    main()
