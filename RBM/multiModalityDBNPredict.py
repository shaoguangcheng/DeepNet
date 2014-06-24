#/usr/bin/env python

import numpy as np

import DBNPredict
import rbmPredict

import sys

def multiModalityDBNPredict(models, X) :
    """
    models : trained models including joint layer
    X : input data for all modalities"""
    N = len(X)

    if N != len(models)-1 :
        print "error"
        sys.exit()

    for index in range(N) :
        X[index] = DBNPredict.DBNPredict(models[index], X[index], isSingleDBN=False)

    # concatenate all modalities data
    for index in range(N-1) :
        if index == 0 :
            data = np.append(X[index], X[index+1], axis = 1)
        else :
            data = np.append(data, X[index+1], axis = 1)

    [prediction, F] = rbmPredict.rbmPredict(models[-1], data)

    return [prediction, F]

if __name__ == "__main__" :
    pass