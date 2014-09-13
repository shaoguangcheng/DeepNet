#!/usr/bin python

import cudamat as cm
import numpy as np

def rbmVtoH(m, X) :
    """convey data fron visual layer to hidden layer"""
    cm.cublas_init()

    # copy data to GPU
    data = cm.CUDAMatrix(cm.reformat(X))
    weight = cm.CUDAMatrix(cm.reformat(m.weight))
    biasH = cm.CUDAMatrix(cm.reformat(m.biasH))

    nCase = X.shape[0]
    nHid = biasH.asarray().size

    hidActP = cm.CUDAMatrix(np.zeros((nCase, nHid)))

    if m.type == "BB" :
        cm.dot(data, weight, target = hidActP)
        hidActP.add_row_vec(biasH)
        hidActP.apply_sigmoid()
    elif m.type == "BG" :
        cm.dot(data, weight, target = hidActP)
        hidActP.add_row_vec(biasH)
    elif m.type == "GB" :
        pass

    result = hidActP.asarray()

    # free device memory
    data.free_device_memory()

    weight.free_device_memory()
    biasH.free_device_memory()
    hidActP.free_device_memory()

    cm.shutdown()

    return result
