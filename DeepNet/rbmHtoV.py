#!/usr/bin python

import cudamat as cm
import numpy as np

def rbmHtoV(m, X) :
    """convey data fron hidden layer to visible layer"""
    cm.cublas_init()

    # copy data to GPU
    data = cm.CUDAMatrix(cm.reformat(X))
    weight = cm.CUDAMatrix(cm.reformat(m.weight))
    biasV = cm.CUDAMatrix(cm.reformat(m.biasV))

    nCase = X.shape[0]
    nVis = biasV.asarray().size
    VisActP = cm.CUDAMatrix(np.zeros((nCase, nVis)))

    if m.type == "BB" :
        cm.dot(data, weight.T, target = VisActP)
        VisActP.add_row_vec(biasV)
        VisActP.apply_sigmoid()
    elif m.type == "BG" :
        cm.dot(data, weight.T, target = VisActP)
        VisActP.add_row_vec(biasV)
    elif m.type == "GB" :
        pass

    result = VisActP.asarray()

    #free device memory
    data.free_device_memory()

    weight.free_device_memory()
    biasV.free_device_memory()
    VisActP.free_device_memory()

    cm.shutdown()

    return result
