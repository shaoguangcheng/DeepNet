#!/usr/bin/env python

import numpy as np

import DBNFit
import DBNPredict
import util
import shelve
import sys


def testDBN(opts) :
    """show how to use DBN to do classification"""
    # read data
    data  = np.load(opts.feature)
    label = np.load(opts.label)

    # set the nodes of hidden layers
    nHid = [5000, 2000]

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

    # set parameters
    # layer1
    p1 = {"maxEpoch" : opts.maxEpoch, "modelType" : "BB"}

    # layer2
    p2 = {"maxEpoch" : opts.maxEpoch}

    p = {"layer1" : p1, "layer2" : p2}

    # train the DBN model
    model = DBNFit.DBNFit(trainData, trainLabel, nHid, name=opts.model, isSingleDBN=True, **p)

    # do prediction for training set and testing set
    [trainR, F1] = DBNPredict.DBNPredict(model, trainData, isSingleDBN=True)
    [testR, F2]  = DBNPredict.DBNPredict(model, example, isSingleDBN=True)

    # calculate classification accuracy
    trainK = 0
    for x in range(nTrain) :
        if trainLabel[x] != trainR[x] :
            trainK = trainK+1

    testK = 0
    for x in range(nTest) :
        if testLabel[x] != testR[x] :
            testK = testK+1

    print "---------------------------------------"
    print "train classification rate : %f " % (1 - trainK*1.0/nTrain)
    print "test  classification rate : %f " % (1 - testK*1.0/nTest)
    print "---------------------------------------"

    if opts.isSaveResult :
        result = shelve.open(opts.resultName)
        result["nHid"]     = nHid
        result["maxEpoch"] = opts.maxEpoch
        result["trainPercent"] = opts.trainPercent
        result["trainAcc"] = 1-trainK*1.0/nTrain
        result["testAcc"]  = 1-testK*1.0/nTest
        result["trainLabel"] = trainLabel
        result["trainR"] = trainR
        result["testLabel"] = testLabel
        result["testR"] = testR
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
                      default='mnist_fea.npy',
                      help="Feature file name")
    parser.add_option("-l", "--label",
                      action="store",
                      dest="label",
                      default='mnist_lab.npy',
                      help="Label file name")
    parser.add_option("-m", "--model",
                      action="store",
                      dest="model",
                      default='DBNModel.npy',
                      help="DBN model file name")
    parser.add_option("-b", "--verbose",
                      action = "store",
                      dest="isSaveResult",
                      default="True",
                      help="whether to save classification result or not")
    parser.add_option("-n", "--name",
                      action="store",
                      dest="resultName",
                      default="classification.shelve",
                      help="the file name of classification result, only works when -b is true")

    (opts, args) = parser.parse_args(argv)

    print("---------------------------------------------------------------")
    print("opts = %s" % str(opts))
    print("args = %s" % str(args))
    print("---------------------------------------------------------------")
    print("")
    return (opts, args)

if __name__ == "__main__" :
    # parse args, opts
    (opts, args) = parseOptions(sys.argv)

    # perform DBN train/test
    testDBN(opts)

