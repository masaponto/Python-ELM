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

    """ ELM model Binary class classification
    """

    def __init__(self, hid_num, a=1):
        """
        Args:
        hid_num (int): number of hidden layer
        a (int) : const value of sigmoid funcion

        """
        self.hid_num = hid_num
        self.a = a  # sigmoid constant value

    def __sigmoid(self, x):
        """sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid

        """

        return 1 / (1 + np.exp(- self.a * x))

    def __G(self, a_v, x_v):
        """output hidden nodes

        Args:
        a_v ([float]): weight vector of hidden layer
        x_v ([float]): input vector

        Returns:
        float: output hidden nodes

        """

        return self.__sigmoid(np.dot(a_v, x_v))

    def __f(self, x_v):
        """output of NN
        Args:
        x_v ([float]): input vector

        Returns:
        int: labels

        """

        return np.sign(np.dot(self.beta_v, [self.__G(a_v, x_v) for a_v in self.a_vs]))

    def _get_hid_matrix(self, x_vs):
        """ output matrix hidden layer
        Args:
        x_vs ([[float]]): input vector

        Returns:
        [[float]]: output matrix of hidden layer

        """

        return np.array([[self.__G(a_v, x_v) for a_v in self.a_vs] for x_v in x_vs])

    def fit(self, X, y):
        """ learning

        Args:
        X [[float]]: feature vectors of learnig data
        y [float] : labels of leanig data

        """

        x_vs = np.array(list(map(self._add_bias, X)))

        # weight hid layer
        np.random.seed()
        self.a_vs = np.random.uniform(-1.0, 1.0, (self.hid_num, len(x_vs[0])))

        # output matrix hidden nodes
        h = self._get_hid_matrix(x_vs)

        # pseudo-inverse matrix of H
        h_t = np.linalg.pinv(h)

        self.out_num = max(y)  # number of class, number of output neuron

        if (self.out_num == 1):
            t_vs = y
            # weight out layer
            self.beta_v = np.dot(h_t, t_vs)

        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            # weight out layer
            self.beta_v = np.transpose(np.dot(h_t, t_vs))

    def _add_bias(self, vec):
        """add bias to list

        Args:
        vec [float]: vec to add bias

        Returns:
        [float]: added vec

        """

        return np.append(vec, 1)

    def predict(self, X):
        """return classify result

        Args:
        X [[float]]: feature vectors of learnig data


        Returns:
        [int]: labels of classification result

        """

        X = np.array(list(map(self._add_bias, X)))
        return np.array([self.__vtol(self.__f(xs)) for xs in X])

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


class COBELM(ELM):

    def __init__(self, hid_num, a = 1, c = 2 ** 0):
        super(COBELM, self).__init__(hid_num, a)
        self.c = c

    def fit(self, X, y):
        """ learning

        Args:
        X [[float]]: feature vectors of learnig data
        y [float] : labels of leanig data

        """
        self.out_num = max(y)  # number of class, number of output neuron
        x_vs = np.array(list(map(self._add_bias, X)))

        # weight hid layer
        np.random.seed()
        self.a_vs = np.random.uniform(-1.0, 1.0, (self.hid_num, len(x_vs[0])))

        # output matrix hidden nodes
        h = self._get_hid_matrix(x_vs)
        I = np.matrix(np.identity(len(h)))
        h_t = np.array(np.dot(h.T, np.linalg.inv(
            (I / self.c) + np.dot(h, h.T))))

        if (self.out_num == 1):
            t_vs = y
            self.beta_v = np.dot(h_t, t_vs)

        else:
            t_vs = np.array(list(map(self._ltov(self.out_num), y)))
            # weight out layer
            self.beta_v = np.transpose(np.dot(h_t, t_vs))


def main():

    db_names = ['wine']
    hid_nums = [10, 20, 30, 1000]

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

        print('COBELM')
        for hid_num in hid_nums:
            print(str(hid_num), end=' ')
            e = COBELM(hid_num)
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
