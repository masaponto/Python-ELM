#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extreme Learning Machine
This script is ELM for binary and multiclass classification.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class ELM (BaseEstimator, ClassifierMixin):

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
        self.a = a

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        return 1 / (1 + np.exp(-self.a * x))

    def _add_bias(self, X):
        """add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        Examples:
        >>> e = ELM(10, 3)
        >>> e._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[1., 2., 3., 1.],
               [1., 2., 3., 1.]])
        """

        return np.c_[X, np.ones(X.shape[0])]

    def _ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        Exmples:
        >>> e = ELM(10, 3)
        >>> e._ltov(3, 1)
        [1, -1, -1]
        >>> e._ltov(3, 2)
        [-1, 1, -1]
        >>> e._ltov(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learnig data
        y [[float]] array : labels of leanig data
        """
        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._ltov(self.out_num, _y) for _y in y])

        # add bias to feature vectors
        X = self._add_bias(X)

        # generate weights between input layer and hidden layer
        np.random.seed()
        self.W = np.random.uniform(-1., 1.,
                                   (self.hid_num, X.shape[1]))

        # find inverse weight matrix
        _H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))

        self.beta = np.dot(_H.T, y)

        return self

    def predict(self, X):
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learnig data

        Returns:
        [int]: labels of classification result
        """
        _H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(_H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


def main():
    from sklearn import preprocessing
    from sklearn.datasets import fetch_openml as fetch_mldata
    from sklearn.model_selection import ShuffleSplit, KFold, cross_val_score

    db_name = 'australian'
    hid_nums = [100, 200, 300]

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)
    data_set.target = [1 if i == 1 else -1
                       for i in  data_set.target.astype(int)]

    for hid_num in hid_nums:
        print(hid_num, end=' ')
        e = ELM(hid_num)

        ave = 0
        for i in range(10):
            cv = KFold(n_splits=5, shuffle=True)
            scores = cross_val_score(
                e, data_set.data, data_set.target,
                cv=cv, scoring='accuracy', n_jobs=-1)
            ave += scores.mean()

        ave /= 10

        print("Accuracy: %0.3f " % (ave))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
