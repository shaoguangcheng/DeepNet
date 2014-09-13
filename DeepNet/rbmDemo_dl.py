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
    nHid = 1000

    # split data and label into  train dataset and test dataset
    trainData  = np.load(opts.trainFeature)
    trainLabel = np.load(opts.trainLabel)
    testData   = np.load(opts.testFeature)
    testLabel  = np.load(opts.testLabel)

    nTrain = trainLabel.size
    nTest  = testLabel.size

    p = {"maxEpoch" : options.maxEpoch}

    m = rbmFit.rbmFit(trainData, nHid, trainLabel, isSaveModel=True, name=opts.model, **p)
    
    [trainR, F1] = rbmPredict.rbmPredict(m, trainData)
    [testR, F2] = rbmPredict.rbmPredict(m, testData)
	
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

    if opts.isSaveResult :
        result = shelve.open(options.resultName)
        result["nHid"]     = nHid
        result["maxEpoch"] = options.maxEpoch
        result["trainAcc"] = 1-trainK*1.0/nTrain
        result["testAcc"]  = 1-testK*1.0/nTest
        result.close()

def parseOptions(argv) :
    """parse arguments"""
    parser = OP(usage="%prog [options] args")

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

    (options, args) = parser.parse_args(argv)

    print "-------------------------------------"
    print "options : ", options
    print "args    : ", args
    print "-------------------------------------"

    return (options, args)

if __name__ == "__main__" :
    (options, argv) = parseOptions(sys.argv)
    testRBM(options)
