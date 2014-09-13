#!/usr/bin/env python
import cudamat as cm
import numpy as np

import sys

import model

def rbmPredict(m, X) :
    """using trained rbm model to do prediction"""
    nClass = m.labels.size
    numCase = X.shape[0]

    # This part is executed on CPU
    # define the free energy
#    FF = np.zeros((numCase, nClass))
#    FFcol = np.zeros((numCase, 1))
#    for index in range(nClass) :
#        temp = np.zeros((numCase, nClass))
#        temp[:, index] = 1
#
#        tt = np.emath.log(np.exp(np.dot(X, m.weight)+ np.dot(temp, m.weightLabel) + m.biasH)+1)
#
#        FFcol = temp[:,index] * m.biasLabel[0,index] + np.sum(tt,axis = 1)
#
#        FF[:, index] = FFcol
#
#    [x, y] = np.where(np.abs(FF - np.max(FF, axis=1, keepdims=True)) < 1e-5)

#    result = np.zeros(y.shape)

#    for index in range(y.size) :
#        result[index] = m.labels[y[index]]


    # The following part runs on GPU
    cm.cublas_init()

    # copy data to GPU
    data = cm.CUDAMatrix(cm.reformat(X))
    weight = cm.CUDAMatrix(cm.reformat(m.weight))
    biasH = cm.CUDAMatrix(cm.reformat(m.biasH))
    weightLabel = cm.CUDAMatrix(cm.reformat(m.weightLabel))
    biasLabel = cm.CUDAMatrix(cm.reformat(m.biasLabel))

    F = cm.CUDAMatrix(np.zeros((numCase, nClass)))
    Fcol = cm.CUDAMatrix(np.zeros((numCase, 1)))
    temp = cm.CUDAMatrix(np.zeros((numCase, nClass)))

    tt = cm.CUDAMatrix(np.zeros((numCase, biasH.asarray().size)))
    for index in range(nClass) :
        temp.assign(0)

        temp.set_col_slice(index, index+1, 1)

        tt = cm.dot(data, weight)
        tt.add_dot(temp, weightLabel)
        tt.add_row_vec(biasH)
        cm.log_1_plus_exp(tt, target = tt, exact = True)

        Fcol = cm.sum(tt, axis = 1)
        Fcol.add_mult(temp.get_col_slice(index, index+1), biasLabel.numpy_array[0, index])

        F.set_col_slice(index, index+1, Fcol)

        tt.free_device_memory()

    F.copy_to_host()
    [x, y] = np.where(np.abs(F.numpy_array - np.max(F.numpy_array, axis=1, keepdims=True)) < 1e-5)

    # free device memory
    data.free_device_memory()

    weight.free_device_memory()
    biasH.free_device_memory()
    biasLabel.free_device_memory()
    weightLabel.free_device_memory()

    F.free_device_memory()
    Fcol.free_device_memory()
    temp.free_device_memory()

    cm.shutdown()

    result = np.zeros(y.shape)

    for index in range(y.size) :
        result[index] = m.labels[y[index]]

    return [result, F.numpy_array]

if __name__ == "__main__" :
    pass
