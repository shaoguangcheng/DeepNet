#!/usr/bin/env python

import random
import sys

import numpy as np
import cudamat as cm

def processOptions(arg, **kwargs) :
    """process options for rbm \
    arg : a dict to save the setting you want to apply to the rbm \
    kwargs : a key word parameter to save the default setting of rbm"""
    result = []

    if kwargs is None :
        kwargs = {}
    kwargs.update(arg)

    return kwargs

def ceil(m ,n) :
    """ m and n must be integer"""
    if  m%n != 0 :
        return m/n+1
    else :
        return m/n

def shuffle(data, label) :
    """shuffle data and label"""
    size = label.shape[0]
    seq = random.sample(range(size), size)

    dataTemp = np.zeros(data.shape)
    labelTemp = np.zeros(label.shape)

    for index, x in enumerate(seq) :
        dataTemp[index, :] = data[x, :]
        labelTemp[index, :] = label[x, :]

    return [dataTemp, labelTemp]

def shuffleMore(data1, data2, label) :
    "shuffle data1 and data2, both have the same corresponding label"
    size = label.shape[0]
    seq = random.sample(range(size), size)

    dataTemp1 = np.zeros(data1.shape)
    dataTemp2 = np.zeros(data2.shape)
    labelTemp = np.zeros(label.shape)

    for index, x in enumerate(seq) :
        dataTemp1[index, :] = data1[x, :]
        dataTemp2[index, :] = data2[x, :]
        labelTemp[index, :] = label[x, :]

    return [dataTemp1, dataTemp2, labelTemp]

def matrixLabel(label) :
    """conver the label format from array to matrix
    label : np.array()"""
    size = label.size

    minLabel = np.min(label)
    maxLabel = np.max(label)

    result = np.zeros((size, maxLabel-minLabel+1))
    for index, x in enumerate(label) :
        result[index, int(x-1)] = 1

    return result

def softmax(prob) :
    """compute the softmax function
    prob : size is batchSize * nclass
        numpy.array format or cm.CUDAMatrix
    softmax function : result(row,col) = exp(prob)/sum_row(exp(prob))"""
    # case cuda matrix
    isCuda = False
    if isinstance(prob, cm.CUDAMatrix) :
        isCuda = True
        prob.copy_to_host()

        # free device memory
        prob.free_device_memory()

    # compute softmax
    prob = np.exp(prob.numpy_array)
    mu = prob/np.sum(prob, axis = 1, keepdims=True)

    # softmax sample
    mu = mu/np.sum(mu, axis = 1, keepdims=True)

    oneofn = np.zeros(mu.shape)
#    [x, y] = np.where(np.abs(mu - np.max(mu, axis=1, keepdims=True)) < 1e-5)
#    oneofn[x, y] = 1

    sample = np.cumsum(mu, axis = 1)
    rows = sample.shape[0]
    sample = sample > np.random.rand(rows,1)

    for index in range(rows) :
        [ix] = np.where(sample[index, :] == True)
        if ix.size == 0 :
            i = 0
        else :
            i = min(ix)

        oneofn[index, i] = 1

    if isCuda :
        oneofn = cm.CUDAMatrix(oneofn)

        return oneofn


def testSoftmax(data) :
    A = cm.CUDAMatrix(data)
    sample = softmax(A)

if __name__ == "__main__" :
    data = np.load(sys.argv[1])
    testSoftmax(data)