#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" ELM

This script is ELM for binary and multiclass classification.
"""

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file


class ELM (BaseEstimator):

    """ 3 step model ELM
    """

    def __init__(self, hid_num, a=1):
        """
        Args:
        hid_num (int): number of hidden layer
        a (int) : const value of sigmoid funcion

        """
        self.hid_num = hid_num
        self.a = a  # sigmoid constant value

    def _sigmoid(self, x):
        """sigmoid function
        Args:
        x ([[float]]) array : input

        Returns:
        float: output of sigmoid

        """

        return 1 / (1 + np.exp(- self.a * x))

    def fit(self, X, y):
        """ learning

        Args:
        X [[float]] array : feature vectors of learnig data
        y [[float]] array : labels of leanig data

        """
        self.out_num = max(y)  # number of class, number of output neuron

        x_vs = self._add_bias(X)

        # weight hid layer
        np.random.seed()
        self.a_vs = np.random.uniform(-1.0, 1.0, (len(x_vs[0]), self.hid_num))

        h_t = np.linalg.pinv(self._sigmoid(np.dot(x_vs, self.a_vs)))

        if (self.out_num == 1):
            self.beta_v = np.dot(h_t, y)

        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            self.beta_v = np.dot(h_t, t_vs)

    def _add_bias(self, x_vs):
        """add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def predict(self, X):
        """return classify result

        Args:
        X [[float]] array: feature vectors of learnig data


        Returns:
        [int]: labels of classification result

        """

        return np.array(list(map(self.__vtol, np.sign(np.dot(self._sigmoid(np.dot(self._add_bias(X), self.a_vs)), self.beta_v)))))

    def __vtol(self, vec):
        """tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> e = ELM(10, 3)
        >>> e.out_num = 3
        >>> e._ELM__vtol([1, -1, -1])
        1
        >>> e._ELM__vtol([-1, 1, -1])
        2
        >>> e._ELM__vtol([-1, -1, 1])
        3
        >>> e._ELM__vtol([-1, -1, -1])
        0

        """

        if self.out_num == 1:
            return round(vec, 0)
        else:
            v = list(vec)
            if len(v) == 1:
                return vec[0]
            elif (max(v) == -1):
                return 0
            return int(v.index(1)) + 1

    def _ltov(self, n):
        """trasform label scalar to vector

            Args:
            n (int) : number of class, number of out layer neuron
            label (int) : label

            Exmples:
            >>> e = ELM(10, 3)
            >>> e._ltov(3)(1)
            [1, -1, -1]
            >>> e._ltov(3)(2)
            [-1, 1, -1]
            >>> e._ltov(3)(3)
            [-1, -1, 1]

            """
        def inltov(label):
            return [-1 if i != label else 1 for i in range(1, n + 1)]
        return inltov


def main():

    db_names = ['australian']
    #db_names = ['iris']

    hid_nums = [10, 20, 30]

    for db_name in db_names:
        print(db_name)
        # load iris data set
        data_set = fetch_mldata(db_name)
        data_set.data = preprocessing.scale(data_set.data)

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
    import doctest
    doctest.testmod()
    main()
