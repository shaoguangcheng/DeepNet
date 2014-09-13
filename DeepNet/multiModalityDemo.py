#!/usr/bin/env python

import sys
from optparse import OptionParser
import shelve

import util
import multiModalityDBNFit
import multiModalityDBNPredict

import numpy as np

def testMultiModalityDBN(opts) :
    """show how to use multi-modality DBN to do classification
    test multi-modality DBN
    In this example, I just use two modalities, but you can extend this code segment to let it work for more modalities"""
    # load data
    viewBasedData = np.load(opts.viewBasedFeature)
    shapeBasedData = np.load(opts.shapeBasedFeature)
    label = np.load(opts.label)

    if viewBasedData.shape[0] != shapeBasedData.shape[0] :
        print "different modalities must have the same number of samples"
        sys.exit()

    nHid = [[1000, 800], [5000, 2000]]
    nJoint = 2800

    # shuffle all data and label
    [viewBasedData, shapeBasedData, label] = util.shuffleMore(viewBasedData, shapeBasedData, label)
    
    percent = opts.trainPercent
    nCase = viewBasedData.shape[0]

    nTrain = int(nCase*percent)
    nTest  = nCase - nTrain

    trainViewBasedData = viewBasedData[0:nTrain, :]
    trainShapeBasedData = shapeBasedData[0:nTrain, :]
    trainLabel = label[0:nTrain, :]

    testViewBasedData  = viewBasedData[nTrain:, :]
    testShapeBasedData  = shapeBasedData[nTrain:, :]
    testLabel = label[nTrain:, :]

    # set parameters for each layer
    # view based layer1
    pV1 = {"maxEpoch" : opts.maxEpoch, "modelType" : "BB"}

    # view based layer2
    pV2 = {"maxEpoch" : opts.maxEpoch}

    p1 = {"layer1" : pV1, "layer2" : pV2}

    # shape based layer1
    pS1 = {"maxEpoch" : opts.maxEpoch, "modelType" : "BB"}

    # shape based layer2
    pS2 = {"maxEpoch" : opts.maxEpoch}

    p2 = {"layer1" : pS1, "layer2" : pS2}

    # joint layer
    pJ = {"maxEpoch" : opts.maxEpoch}

    p = {"modality1" : p1, "modality2" : p2, "joint" : pJ}

    # train the multi-modality model
    model = multiModalityDBNFit.multiModalityDBNFit([trainViewBasedData, trainShapeBasedData],\
                                                    trainLabel, nHid, nJoint, isSaveModel=True, modelName = opts.model, **p)

    # do prediction for training set and testing set
    [trainR, F1] = multiModalityDBNPredict.multiModalityDBNPredict(model, [trainViewBasedData, trainShapeBasedData])
    [testR, F2] = multiModalityDBNPredict.multiModalityDBNPredict(model, [testViewBasedData, testShapeBasedData])

    # calculate the classification accuracy
    trainK = 0
    for x in range(nTrain) :
        if trainLabel[x] != trainR[x] :
            trainK = trainK+1

    testK = 0
    for x in range(nTest) :
        if testLabel[x] != testR[x] :
            testK = testK+1

    print "---------------------------------------"
    print "train classification rate : %f " % (1-trainK*1.0/nTrain)
    print "test  classification rate : %f " % (1-testK*1.0/nTest)
    print "---------------------------------------"

    if opts.isSaveResult :
        result = shelve.open(opts.resultName)
        result["nHid"]     = nHid
        result["nJoint"]   = nJoint
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
    parser = OptionParser(usage="usage: %prog [options] args",
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
                      default=2000,
                      type='int',
                      help="Iteration number")

    parser.add_option("-v", "--viewBasedFeature",
                      action="store",
                      dest="viewBasedFeature",
                      default='example/multi-modal_demo/SHREC_2007_BOW_1000_viewBased.npy',
                      help="Feature file name")
    parser.add_option("-s", "--shapeBasedFeature",
                      action="store",
                      dest="shapeBasedFeature",
                      default='example/multi-modal_demo/SHREC_2007_BOW_100_shapeBased.npy',
                      help="Feature file name")
    parser.add_option("-l", "--label",
                      action="store",
                      dest="label",
                      default='example/multi-modal_demo/label.npy',
                      help="Label file name")
    parser.add_option("-m", "--model",
                      action="store",
                      dest="model",
                      default='example/multi-modal_demo/multiModalityModel.npy',
                      help="multi modality model file name to save")
    parser.add_option("-b", "--verbose",
                      action = "store",
                      dest="isSaveResult",
                      default="True",
                      help="whether to save classification result or not")
    parser.add_option("-n", "--name",
                      action="store",
                      dest="resultName",
                      default="example/multi-modal_demo/classification.shelve",
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
    testMultiModalityDBN(opts)
