#!/usr/bin/python
import cudamat as cm
import numpy as np

import sys

import model as m
import util

def rbm(data, numHid, modelType = "BB", **kwargs) :
    """
    rbm defination
    data : when type is BB, should be binary, or in [0,1] to be interpreted as probabilities
           when type is GB, should be continuous real value. data should have a format of *.npy
    numHid : number nodes of and hidden layer
    type   : rbm type, can be set as BB or GB

additional inputs (specified as name value pairs or in struct)
    method          CD or SML
    eta             learning rate
    momentum        momentum for smoothness amd to prevent overfitting
                    NOTE: momentum is not recommended with SML
    maxepoch        # of epochs: each is a full pass through train data
    avglast         how many epochs before maxepoch to start averaging
                before. Procedure suggested for faster convergence by
                Kevin Swersky in his MSc thesis
    penalty         weight decay factor
    batchsize       The number of training instances per batch
    verbose         For printing progress
    anneal          Flag. If set true, the penalty is annealed linearly
                through epochs to 10% of its original value

    OUTPUTS:
    model.type      Type of RBM (i.e. type of its visible and hidden units)
    model.weight         The weights of the connections
    model.biasH         The biases of the hidden layer
    model.biasV         The biases of the visible layer
    model.top       The activity of the top layer, to be used when training
                    DBN's
    errors          The errors in reconstruction at every epoch
       """

    arg = util.processOptions(kwargs, \
                            method = "CD", \
                            eta = 0.1, \
                            momentum = 0.9,\
                            maxEpoch = 50, \
                            avgLast = 0, \
                            penalty = 0, \
                            batchSize = 50, \
                            verbose = True, \
                            anneal = False)
    [method, eta, momentum, maxEpoch, avgLast, penalty, batchSize, verbose, anneal] = [\
        arg["method"],\
        arg["eta"],\
        arg["momentum"],\
        arg["maxEpoch"],\
        arg["avgLast"],\
        arg["penalty"],\
        arg["batchSize"],\
        arg["verbose"],\
        arg["anneal"]
    ]

    # from which step, we start to compute the average
    avgStart = maxEpoch - avgLast

    # for weight decay use
    oldPenalty = penalty

    # numCases : number of example
    # numDims : the length of each example
    # each row is an example
    [numCases, numDims] = list(data.shape)

    if verbose :
        print "processing data"

    numVis = numDims
    numBatch = util.ceil(numCases,batchSize)

    # shuffle the data
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
    hidActP2 = cm.empty((batchSize, numHid))
    visState = cm.empty((batchSize,numVis))
    hidState = cm.empty((batchSize, numHid))

    t = 1
    for epoch in range(maxEpoch) :
        error = []

        if anneal :
            # apply linear weight decay
            penalty = oldPenalty - 0.9 *epoch/maxEpoch*oldPenalty

        for batch in range(numBatch) :
            # train each data batch
            if batchSize*(batch+1) > numCases :
                visTrue = deviceData.get_row_slice(batchSize*batch, numCases)
                batchSize = visTrue.shape[0]
            else :
                visTrue = deviceData.get_row_slice(batchSize*batch, batchSize*(batch+1))
                batchSize = visTrue.shape[0]

            visActP.assign(visTrue)

            # positive phase
            cm.dot(visActP, weight, target = hidActP)
            hidActP.add_row_vec(biasH)
            hidActP.apply_sigmoid()

            hidState.fill_with_rand()
            hidState.less_than(hidActP, target=hidState)

            if cmp(method, "SML") == 0 :
                if np.logical_and(np.equal(epoch,1), np.equal(batch,1)) :
                    pass # here does not need in practical use
            elif cmp(method, "CD") == 0 :
                pass

            # negetive phase
            if cmp(modelType, "BB") == 0 :
                cm.dot(hidState, weight.transpose(), target = visActP)
                visActP.add_row_vec(biasV)
                visActP.apply_sigmoid()

                visState.fill_with_rand()
                visState.less_than(visActP, target = visState)
            elif cmp(modelType, "GB") == 0 :
                cm.dot(hidState, weight.transpose(), target = visActP)
                visActP.add_row_vec(biasV)

                visActP.add(np.random.randn(batchSize, numVis),target=visState)

            # another positive phase
            cm.dot(visState, weight, target = hidActP2)
            hidActP2.add_row_vec(biasH)
            hidActP2.apply_sigmoid()

            hidState.fill_with_rand()
            hidState.less_than(hidActP2, target=hidState)

            #update weight and bias
            dWeight = cm.dot(visTrue.transpose(), hidActP)
            dWeight.subtract_dot(visState.transpose(), hidActP2)
            dBiasV = visTrue.sum(axis = 0).subtract(visState.sum(axis = 0))
            dBiasH = hidActP.sum(axis=0).subtract(hidActP2.sum(axis = 0))

            dWeight.divide(batchSize).subtract(weight.mult(penalty))
            dBiasV.divide(batchSize)
            dBiasH.divide(batchSize)

            weightInc.mult(momentum).add_mult(dWeight, eta)
            biasVInc.mult(momentum).add_mult(dBiasV, eta)
            biasHInc.mult(momentum).add_mult(dBiasH, eta)

            weight.add(weightInc)
            biasV.add(biasVInc)
            biasH.add(biasHInc)

            if epoch > avgStart :
                # apply average
                weightAgv.subtract(weightAgv.subtract(weight).mult(1.0/t))
                biasVAgv.subtract(biasVAgv.subtract(biasV).mult(1.0/t))
                biasHAgv.subtract(biasHAgv.subtract(biasH).mult(1.0/t))
                t = t+1
            else :
                weightAgv = weight
                biasVAgv = biasV
                biasHAgv = biasH

            # reconstruction error
            visTrue.subtract(visActP)
            error.append(visTrue.euclid_norm() ** 2)
        if verbose :
            print "epoch %d/%d. Reconstruction error is %f " % (epoch+1, maxEpoch, sum(error))

    # save rbm model
    top = cm.CUDAMatrix(np.zeros((numCases, numHid)))
    cm.dot(deviceData, weightAgv, target = top)
    top.add_row_vec(biasHAgv)
    top.apply_sigmoid()

    model_ = m.rbmModel(weightAgv,biasVAgv,biasHAgv,type = modelType,top = top)

    cm.shutdown()

    return model_

if __name__ == "__main__" :
    print sys.argv
    data = np.load(sys.argv[1])
    rbm(data, int(sys.argv[2]))






