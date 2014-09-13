#!/usr/bin/env python

import sys
from optparse import OptionParser as OP

import rbmFit
import rbmPredict
import util
import shelve

import numpy as np

def testRBM(opts) :
    """show how to use RBM to do classification"""
    # read data
    data  = np.load(opts.feature)
    label = np.load(opts.label)

    # set the nodes of hidden layers
    nHid = 1000

    # shuffle data and label
    [data, label] = util.shuffle(data, label)

    # decide how many samples to be used as training set
    percent = float(opts.trainPercent)
    nCase   = data.shape[0]

    nTrain = int(nCase * percent)
    nTest = nCase - nTrain

    # split data and label into  train dataset and test dataset
    trainData  = data[0:nTrain, :]
    trainLabel = label[0:nTrain, :]
    example   = data[nTrain:, :]
    testLabel  = label[nTrain:, :]

    p = {"maxEpoch" : opts.maxEpoch}

    m = rbmFit.rbmFit(trainData, nHid, trainLabel, isSaveModel=True, name=opts.model, **p)
    
    [trainR, F1] = rbmPredict.rbmPredict(m, trainData)
    [testR, F2] = rbmPredict.rbmPredict(m, example)
	
    trainK = 0
    for x in range(nTrain) :
        if trainLabel[x] != trainR[x] :
            trainK = trainK + 1

    testK = 0
    for x in range(nTest) :
        if testLabel[x] != testR[x] :
            testK = testK+1

    print "---------------------------------------"
    print "train classification rate : %f " % (1-trainK*1.0/nTrain)
    print "test  classification rate : %f " % (1-testK*1.0/nTest)
    print "---------------------------------------"

    if options.isSaveResult :
        result = shelve.open(options.resultName)
        result["nHid"]     = nHid
        result["maxEpoch"] = options.maxEpoch
        result["trainAcc"] = 1-trainK*1.0/nTrain
        result["testAcc"]  = 1-testK*1.0/nTest
        result.close()

def parseOptions(argv):
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] filenames",
                          version="%prog 1.0")

    parser.add_option("-p", "--trainPercent",
                      action="store",
                      dest="trainPercent",
                      default=0.5,
                      type='float',
                      help="Trainning data percentage")
    parser.add_option("-e", "--maxEpoch",
                      action="store",
                      dest="maxEpoch",
                      default=1000,
                      type='int',
                      help="Iteration number")
                      
    parser.add_option("-f", "--feature",
                      action="store",
                      dest="feature",
                      default='example/mnist_fea.npy',
                      help="Feature file name")
    parser.add_option("-l", "--label",
                      action="store",
                      dest="label",
                      default='example/mnist_lab.npy',
                      help="Label file name")
    parser.add_option("-m", "--model",
                      action="store",
                      dest="model",
                      default='example/DBNModel.npy',
                      help="DBN model file name")
    parser.add_option("-b", "--verbose",
                      action = "store",
                      dest="isSaveResult",
                      default="True",
                      help="whether to save classification result or not")
    parser.add_option("-n", "--name",
                      action="store",
                      dest="resultName",
                      default="example/classification.shelve",
                      help="the file name of classification result, only works when -b is true")

    (opts, args) = parser.parse_args(argv)

    print("---------------------------------------------------------------")
    print("opts = %s" % str(opts))
    print("args = %s" % str(args))
    print("---------------------------------------------------------------")
    print("")
    return (opts, args)

if __name__ == "__main__" :
    (options,args) = parseOptions(sys.argv)
    testRBM(options)
