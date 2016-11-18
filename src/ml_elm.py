#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from elm import ELM


class MLELM(ELM):
    """
    Multi Layer Extreme Learning Machine

    """

    def __init__(self, hidden_units, a=1):
        self.hidden_units = hidden_units
        self.betas = []
        self.a = a

    def __calc_hidden_layer(self, X):
        """
        Args:
        X np.array input feature vector
        """
        for beta in self.betas:
            X = np.dot(beta, X.T).T
        return X

    def fit(self, X, y):
        self.out_num = max(y)
        X = self._add_bias(X)

        for hid_num in self.hidden_units[:-1]:
            _X = self.__calc_hidden_layer(X)
            np.random.seed()
            W = np.random.uniform(-1., 1.,
                                  (hid_num, _X.shape[1]))
            _H = np.linalg.pinv(self._sigmoid(np.dot(W, _X.T)))
            beta = np.dot(_H.T, _X)
            self.betas.append(beta)

        _X = self.__calc_hidden_layer(X)

        self.elm = ELM(hid_num=self.hidden_units[-1])
        self.elm.fit(_X, y)

        return self

    def predict(self, X):
        X = self.__calc_hidden_layer(self._add_bias(X))
        return self.elm.predict(X)


def main():
    from sklearn import preprocessing
    from sklearn.datasets import fetch_mldata
    from sklearn.model_selection import train_test_split

    db_name = 'diabetes'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)

    X_train, X_test, y_train, y_test = train_test_split(
        data_set.data, data_set.target, test_size=0.4)

    mlelm = MLELM(hidden_units=(10, 30, 200)).fit(X_train, y_train)
    elm = ELM(200).fit(X_train, y_train)

    print("MLELM Accuracy %0.3f " % mlelm.score(X_test, y_test))
    print("ELM Accuracy %0.3f " % elm.score(X_test, y_test))


if __name__ == "__main__":
    main()
