#!/usr/bin/env python

import cudamat as cm
import numpy as np

import sys
import copy

import util
import model as m

def rbmFit(X, numHid, y, isSaveModel=False, name=None, **kwargs) :
    """
    X              ... data. should be binary, or in [0,1] interpreted as
                   ... probabilities
    numhid         ... number of hidden units
    y              ... List of discrete labels

    nClass          number of classes
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

    model.weight         The weights of the connections
    model.biasH         The biases of the hidden layer
    model.biasV         The biases of the visible layer

    model.weightlabel       ... The weights on labels layer
    model.biasLabel       ... The biases on labels layer

    errors          The errors in reconstruction at each epoch
       """

    arg = util.processOptions(kwargs, \
                            nClass = np.unique(y).size, \
                            method = "CD", \
                            eta = 0.1, \
                            momentum = 0.5,\
                            maxEpoch = 500, \
                            avgLast = 0, \
                            penalty = 0, \
                            batchSize = 100, \
                            verbose = True)
    [nClass, method, eta, momentum, maxEpoch, avgLast, penalty, batchSize, verbose] = [\
        arg["nClass"],\
        arg["method"],\
        arg["eta"],\
        arg["momentum"],\
        arg["maxEpoch"],\
        arg["avgLast"],\
        arg["penalty"],\
        arg["batchSize"],\
        arg["verbose"]
    ]

    if verbose :
        print "Processing data ..."

    # from which step, we start to compute the average
#    avgStart = maxEpoch - avgLast

    # for weight decay use
#    oldPenalty = penalty

    # numCases : number of example
    # numDims : the length of each example
    # each row is an example
    [numCases, numDims] = list(X.shape)

    numVis = numDims
    uniqueLabel = np.unique(y)
    numBatch = util.ceil(numCases, batchSize)

    y = util.matrixLabel(y)

    # shuffle data and label
    data = copy.deepcopy(X)
    [data, label] = util.shuffle(data, y)

    # init CUDA
    cm.cublas_init()
    cm.CUDAMatrix.init_random(100)
    deviceData = cm.CUDAMatrix(cm.reformat(data))
    deviceLabel = cm.CUDAMatrix(cm.reformat(label))

    # init weights
    weight = cm.CUDAMatrix(0.1*np.random.randn(numVis,numHid))
    biasV = cm.CUDAMatrix(np.zeros((1, numVis)))
    biasH = cm.CUDAMatrix(np.zeros((1, numHid)))
    weightLabel = cm.CUDAMatrix(0.1*np.random.randn(nClass, numHid))
    biasLabel = cm.CUDAMatrix(np.zeros((1,nClass)))

    # init weight update
    weightInc = cm.CUDAMatrix(np.zeros((numVis,numHid)))
    biasVInc = cm.CUDAMatrix(np.zeros((1,numVis)))
    biasHInc = cm.CUDAMatrix(np.zeros((1,numHid)))
    weightLabelInc = cm.CUDAMatrix(np.zeros((nClass, numHid)))
    biasLabelInc = cm.CUDAMatrix(np.zeros((1,nClass)))

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
                labelTrue = deviceLabel.get_row_slice(batchSize*batch, numCases)
                batchSize = visTrue.shape[0]

                visActP = cm.empty((batchSize, numVis))
                hidActP = cm.empty((batchSize, numHid))
                hidState = cm.empty((batchSize, numHid))
            else :
                visTrue = deviceData.get_row_slice(batchSize*batch, batchSize*(batch+1))
                labelTrue = deviceLabel.get_row_slice(batchSize*batch, batchSize*(batch+1))
                batchSize = visTrue.shape[0]

            visActP.assign(visTrue)

            #apply momentum
            weightInc.mult(momentum)
            biasVInc.mult(momentum)
            biasHInc.mult(momentum)
            weightLabel.mult(momentum)
            biasLabel.mult(momentum)

            # positive phase
            cm.dot(visActP, weight, target = hidActP)
            hidActP.add_dot(labelTrue, weightLabel)
            hidActP.add_row_vec(biasH)
            hidActP.apply_sigmoid()

            weightInc.add_dot(visActP.T, hidActP)
            biasVInc.add_sums(visActP, axis=0)
            biasHInc.add_sums(hidActP, axis=0)
            weightLabelInc.add_dot(labelTrue.T, hidActP)
            biasLabelInc.add_sums(labelTrue, axis=0)

            hidState.fill_with_rand()
            hidState.less_than(hidActP, target=hidActP)

            if cmp(method, "SML") == 0 :
                if np.logical_and(np.equal(epoch,1), np.equal(batch,1)) :
                    pass # here does not need in practical use
            elif cmp(method, "CD") == 0 :
                pass

            # negative phase
            cm.dot(hidActP, weight.T, target = visActP)
            visActP.add_row_vec(biasV)
            visActP.apply_sigmoid()

            cm.dot(hidActP, weightLabel.T, target = labelTrue)
            labelTrue.add_row_vec(biasLabel)
            labelTrue = util.softmax(labelTrue)

            # another positive phase
            cm.dot(visActP, weight, target = hidActP)
            hidActP.add_dot(labelTrue, weightLabel)
            hidActP.add_row_vec(biasH)
            hidActP.apply_sigmoid()

            weightInc.subtract_dot(visActP.T, hidActP)
            biasVInc.add_sums(visActP, axis=0, mult=-1)
            biasHInc.add_sums(hidActP, axis=0, mult=-1)
            weightLabelInc.subtract_dot(labelTrue.T, hidActP)
            biasLabelInc.add_sums(labelTrue, axis=0, mult=-1)

            # update weights and bias
            weight.add_mult(weightInc, eta/batchSize)
            biasV.add_mult(biasVInc, eta/batchSize)
            biasH.add_mult(biasHInc, eta/batchSize)
            weightLabel.add_mult(weightLabelInc, eta/batchSize)
            biasLabel.add_mult(biasLabelInc, eta/batchSize)

            # calculate reconstruction error
            visTrue.subtract(visActP)
            error.append(visTrue.euclid_norm()**2)

            # free memory
            visTrue.free_device_memory()
            labelTrue.free_device_memory()

        if verbose :
            print "Epoch %d/%d, reconstruction error is %f " % (epoch+1, maxEpoch, sum(error))

    # save rbm model
    weight.copy_to_host()
    biasV.copy_to_host()
    biasH.copy_to_host()
    weightLabel.copy_to_host()
    biasLabel.copy_to_host()

    model_ = m.rbmModel(weight.numpy_array, biasV.numpy_array, biasH.numpy_array, \
                        weightLabel = weightLabel.numpy_array,\
                        biasLabel = biasLabel.numpy_array, labels = uniqueLabel)

    # free device memory
    deviceData.free_device_memory()
    deviceLabel.free_device_memory()

    weight.free_device_memory()
    biasV.free_device_memory()
    biasH.free_device_memory()
    weightLabel.free_device_memory()
    biasLabel.free_device_memory()

    weightInc.free_device_memory()
    biasVInc.free_device_memory()
    biasHInc.free_device_memory()
    weightLabelInc.free_device_memory()
    biasLabelInc.free_device_memory()

    hidActP.free_device_memory()
    visActP.free_device_memory()
    hidState.free_device_memory()

    cm.shutdown()

    if isSaveModel :
        modelList = []
        modelList.append(model_)
        model = np.array(modelList)
        np.save(name,model)

    return model_

if __name__ == "__main__" :
    data = np.load(sys.argv[1])
    label = np.load(sys.argv[2])
    rbmFit(data, int(sys.argv[3]), label)