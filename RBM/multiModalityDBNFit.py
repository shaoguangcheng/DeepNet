#/usr/bin/env python

import numpy as np

import DBNFit
import rbmFit

import sys

def multiModalityDBNFit(X, label, numHid, numJoint, isSaveModel=True, modelName = None, **kwargs) :
    """"multi-modality DBN fitting
    X : input data for all modalities
    lable : the label of each sample
    numHid : the node of each hidden layer of each modality and must be a two-dimension array
    numJoint : the node of joint layer(if there is only one modality, numJoint can be seen as the node of last hidden layer)

    for example : multiModalityDBN([img, txt], label, [[300,400], [200,300]])"""

    # cal the total number of modality
    N = len(X)
    if N != len(numHid) :
        print "the X and numHid must have the same length"
        sys.exit()

    models = list()

    # train all modalities
    for index in range(N) :
        string = "modality" + str(index+1)

        # here isSingleDBN must be set to False
        m = DBNFit.DBNFit(X[index], label, numHid[index], isSaveModels=False, name = "./DBN.npy", \
                          isSingleDBN = False, **kwargs[string])
        models.append(m)

    # train the joint layer
    # concatenate all modalities data
#    nDim = 0
#    nCase = label.size
#    for index in range(N) :
#        nDim = nDim + numHid[index][-1]

#    inputData = np.zeros((nCase, nDim))
    for index in range(N-1) :
        if index == 0 :
            data = np.append(models[index][-1].top, models[index+1][-1].top, axis = 1)
        else :
            data = np.append(data, models[index+1][-1].top, axis = 1)

    string = "joint"
    m = rbmFit.rbmFit(data, numJoint, label, **kwargs[string])

    models.append(m)

    if isSaveModel :
        models_ = np.array(models)
        if modelName == None :
            modelName = "../data/model/multi-modalityDBN.npy"

        np.save(modelName, models_)

    return models