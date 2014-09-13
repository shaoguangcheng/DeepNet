#!/usr/bin/env python

import numpy as np

import DBNFit
import DBNPredict
import util
import shelve
import sys


def testDBN(opts) :
    """show how to use DBN to do classification"""
    # set the nodes of hidden layers
    nHid = [1500, 1000]

    # split data and label into  train dataset and test dataset
    trainData  = np.load(opts.trainFeature)
    trainLabel = np.load(opts.trainLabel)
    testData   = np.load(opts.testFeature)
    testLabel  = np.load(opts.testLabel)

	nTrain = trainLabel.size
	ntest  = testLabel.size

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
    [testR, F2]  = DBNPredict.DBNPredict(model, testData, isSingleDBN=True)

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
        result["trainAcc"] = 1-trainK*1.0/nTrain
        result["testAcc"]  = 1-testK*1.0/nTest
        result.close()

def parseOptions(argv):
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] filenames",
                          version="%prog 1.0")

    parser.add_option("-e", "--maxEpoch",
                      action="store",
                      dest="maxEpoch",
                      default=1000,
                      type='int',
                      help="Iteration number")
                      
    parser.add_option("--trainFeature",
                      action="store",
                      dest="trainFeature",
 #                     default='testData/feature.npy',
                      help="trainning Feature file name")
    parser.add_option("--testFeature",
                      action="store",
                      dest="testFeature",
 #                     default='testData/feature.npy',
                      help="testing Feature file name")

    parser.add_option("--trainLabel",
                      action="store",
                      dest="trainLabel",
#                      default='testData/label.npy',
                      help="train Label file name")
    parser.add_option("--testLabel",
                      action="store",
                      dest="testLabel",
#                      default='testData/label.npy',
                      help="test Label file name")
    parser.add_option("--model",
                      action="store",
                      dest="model",
#                      default='testData/DBNModel.npy',
                      help="DBN model file name")
    parser.add_option("--verbose",
                      action = "store",
                      dest="isSaveResult",
                      default="True",
                      help="whether to save classification result or not")
    parser.add_option( "--name",
                      action="store",
                      dest="resultName",
#                      default="testData/classification.shelve",
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

