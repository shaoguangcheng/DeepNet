#!/usr/bin/env python
import cudamat as cm
import numpy as np

import sys
import copy

import model as m
import util

def rbm(X, numHid, **kwargs) :
    """
    rbm defination
    data : when type is BB, should be binary, or in [0,1] to be interpreted as probabilities
           when type is GB, should be continuous real value. data should have a format of *.npy
    numHid : number nodes of and hidden layer
    type   : rbm type, can be set as BB or GB

    method          CD or SML
    eta             learning rate
    momentum        momentum for smoothness amd to prevent overfitting
                    NOTE: momentum is not recommended with SML
    maxepoch        # of epochs: each is a full pass through train data
    avglast         how many epochs before maxepoch to start averaging
                before. Procedure suggested for faster convergence by
                Kevin Swersky in his MSc thesis

    batchsize       The number of training instances per batch
    verbose         For printing progress

    model.type      Type of RBM (i.e. type of its visible and hidden units)
    model.weight         The weights of the connections
    model.biasH         The biases of the hidden layer
    model.biasV         The biases of the visible layer
    model.top       The activity of the top layer, to be used when training
                    DBN's
    errors          The errors in reconstruction at every epoch
       """
# when compute the transpose of a matrix, using the method *.transpose() is much space consuming. I suggest we can use
#  .T atrribute instead

    arg = util.processOptions(kwargs, \
                            modelType = "BB", \
                            method = "CD", \
                            eta = 0.1, \
                            momentum = 0.5,\
                            maxEpoch = 500, \
                            avgLast = 0, \
                            penalty = 0, \
                            batchSize = 100, \
                            verbose = True)

    [modelType, method, eta, momentum, maxEpoch, avgLast, penalty, batchSize, verbose] = [\
        arg["modelType"], \
        arg["method"],\
        arg["eta"],\
        arg["momentum"],\
        arg["maxEpoch"],\
        arg["avgLast"],\
        arg["penalty"],\
        arg["batchSize"],\
        arg["verbose"]
    ]

    # from which step, we start to compute the average
#    avgStart = maxEpoch - avgLast

    # for weight decay use
#    oldPenalty = penalty

    # numCases : number of example
    # numDims : the length of each example
    # each row is an example
    [numCases, numDims] = list(X.shape)

    if verbose :
        print "processing data"

    numVis = numDims
    numBatch = util.ceil(numCases, batchSize)

    # shuffle the data
    data = copy.deepcopy(X)
    np.random.shuffle(data)

    # init CUDA
#    cm.cuda_set_device()
    cm.cublas_init()
    cm.CUDAMatrix.init_random(100)
    deviceData = cm.CUDAMatrix(cm.reformat(data))

    # init weights
    weight = cm.CUDAMatrix(0.1*np.random.randn(numVis,numHid))
    biasV = cm.CUDAMatrix(np.zeros((1, numVis)))
    biasH = cm.CUDAMatrix(np.zeros((1, numHid)))

    # init weight update
    weightInc = cm.CUDAMatrix(np.zeros((numVis,numHid)))
    biasVInc = cm.CUDAMatrix(np.zeros((1,numVis)))
    biasHInc = cm.CUDAMatrix(np.zeros((1,numHid)))

    #init temporary storage
    visActP = cm.empty((batchSize, numVis))
    hidActP = cm.empty((batchSize, numHid))
    hidState = cm.empty((batchSize, numHid))

    for epoch in range(maxEpoch) :
        error = []

        for batch in range(numBatch) :
            # train each data batch
            if batchSize*(batch+1) > numCases :
                visTrue = deviceData.get_row_slice(batchSize*batch, numCases)
                batchSize = visTrue.shape[0]

                visActP = cm.empty((batchSize, numVis))
                hidActP = cm.empty((batchSize, numHid))
                hidState = cm.empty((batchSize, numHid))
            else :
                visTrue = deviceData.get_row_slice(batchSize*batch, batchSize*(batch+1))
                batchSize = visTrue.shape[0]

            visActP.assign(visTrue)

            #apply momentum
            weightInc.mult(momentum)
            biasVInc.mult(momentum)
            biasHInc.mult(momentum)

            # positive phase
            cm.dot(visActP, weight, target = hidActP)
            hidActP.add_row_vec(biasH)
            hidActP.apply_sigmoid()

            weightInc.add_dot(visActP.T, hidActP)
            biasVInc.add_sums(visActP, axis=0)
            biasHInc.add_sums(hidActP, axis=0)

            hidState.fill_with_rand()
            hidState.less_than(hidActP, target=hidActP)

            if cmp(method, "SML") == 0 :
                if np.logical_and(np.equal(epoch,1), np.equal(batch,1)) :
                    pass # here does not need in practical use
            elif cmp(method, "CD") == 0 :
                pass

            # negetive phase
            if cmp(modelType, "BB") == 0 :
                cm.dot(hidActP, weight.T, target = visActP)
                visActP.add_row_vec(biasV)
                visActP.apply_sigmoid()

            elif cmp(modelType, "GB") == 0 :
                cm.dot(hidActP, weight.T, target = visActP)
                visActP.add_row_vec(biasV)

                visActP.add(np.random.randn(batchSize, numVis),target=visActP)

            # another positive phase
            cm.dot(visActP, weight, target = hidActP)
            hidActP.add_row_vec(biasH)
            hidActP.apply_sigmoid()

            weightInc.subtract_dot(visActP.T, hidActP)
            biasVInc.add_sums(visActP, axis=0, mult=-1)
            biasHInc.add_sums(hidActP, axis=0, mult=-1)

            #update weight and bias
            weight.add_mult(weightInc, eta/batchSize)
            biasV.add_mult(biasVInc, eta/batchSize)
            biasH.add_mult(biasHInc, eta/batchSize)

#            if epoch > avgStart :
#                # apply average
#                weightAgv.subtract(weightAgv.subtract(weight).mult(1.0/t))
#                biasVAgv.subtract(biasVAgv.subtract(biasV).mult(1.0/t))
#                biasHAgv.subtract(biasHAgv.subtract(biasH).mult(1.0/t))
#                t = t+1
#            else :
#                weightAgv = weight
#                biasVAgv = biasV
#                biasHAgv = biasH

            # reconstruction error
            visTrue.subtract(visActP)
            error.append(visTrue.euclid_norm() ** 2)

            # free device memory
            visTrue.free_device_memory()

        if verbose :
            print "epoch %d/%d. Reconstruction error is %f " % (epoch+1, maxEpoch, sum(error))

    # save rbm model
    top = cm.CUDAMatrix(np.zeros((numCases, numHid)))
    cm.dot(cm.CUDAMatrix(cm.reformat(X)), weight, target = top)
    top.add_row_vec(biasH)
    top.apply_sigmoid()

    weight.copy_to_host()
    biasV.copy_to_host()
    biasH.copy_to_host()
    top.copy_to_host()

    model_ = m.rbmModel(weight.numpy_array, biasV.numpy_array, \
                        biasH.numpy_array, type = modelType, top = top.numpy_array)


    # free device memory
    deviceData.free_device_memory()

    weight.free_device_memory()
    biasV.free_device_memory()
    biasH.free_device_memory()

    weightInc.free_device_memory()
    biasVInc.free_device_memory()
    biasHInc.free_device_memory()

    hidActP.free_device_memory()
    visActP.free_device_memory()
    hidState.free_device_memory()

    cm.shutdown()

    return model_

if __name__ == "__main__" :
    data = np.load(sys.argv[1])
    rbm(data, int(sys.argv[2]))





