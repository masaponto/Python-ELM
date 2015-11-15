#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extreme Learning Machine
This script is ELM for binary and multiclass classification.
"""

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file


class ELM (BaseEstimator):

    """
    3 step model ELM
    """

    def __init__(self,
                 hid_num,
                 a=1):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid funcion
        """
        self.hid_num = hid_num
        self.a = a  # sigmoid constant value

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x ([[float]]) array : input

        Returns:
        float: output of sigmoid
        """

        return 1 / (1 + np.exp(- self.a * x))

    def _add_bias(self, x_vs):
        """add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        Examples:
        >>> e = ELM(10, 3)
        >>> e._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[ 1.,  2.,  3.,  1.],
               [ 1.,  2.,  3.,  1.]])
        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def _vtol(self, vec):
        """
        tranceform vector (list) to label

        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> e = ELM(10, 3)
        >>> e.out_num = 3
        >>> e._vtol([1, -1, -1])
        1
        >>> e._vtol([-1, 1, -1])
        2
        >>> e._vtol([-1, -1, 1])
        3
        """

        if self.out_num == 1:
            return round(vec, 0)
        else:
            v = list(vec)
            return int(v.index(max(v))) + 1

    def _ltov(self, n):
        """
        trasform label scalar to vector
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
        def in_ltov(label):
            return [-1 if i != label else 1 for i in range(1, n + 1)]
        return in_ltov

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learnig data
        y [[float]] array : labels of leanig data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        # add bias to feature vectors
        x_vs = self._add_bias(X)

        # generate weights between input layer and hidden layer
        np.random.seed()
        self.a_vs = np.random.uniform(-1., 1.,
                                      (len(x_vs[0]), self.hid_num))

        # find inverse weight matrix
        h_t = np.linalg.pinv(self._sigmoid(np.dot(x_vs, self.a_vs)))

        # find weights between output layer
        if self.out_num == 1:
            self.beta_v = np.dot(h_t, y)
        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            self.beta_v = np.dot(h_t, t_vs)

    def predict(self, X):
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [int]: labels of classification result
        """

        return np.array(list(map(self._vtol, np.sign(np.dot(self._sigmoid(np.dot(self._add_bias(X), self.a_vs)), self.beta_v)))))


def main():

    db_names = ['australian', 'iris']

    hid_nums = [10, 20, 30]

    for db_name in db_names:
        print(db_name)
        data_set = fetch_mldata(db_name)
        data_set.data = preprocessing.scale(data_set.data)

        for hid_num in hid_nums:
            print(hid_num, end=' ')
            e = ELM(hid_num)
            ave = 0
            for i in range(10):
                scores = cross_validation.cross_val_score(
                    e, data_set.data, data_set.target, cv=5, scoring='accuracy')
                ave += scores.mean()
            ave /= 10
            print("Accuracy: %0.3f " % (ave))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
