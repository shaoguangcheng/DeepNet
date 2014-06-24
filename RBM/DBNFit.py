#/usr/bin/env python

import numpy as np
import rbm
import rbmFit

import sys

def DBNFit(X, label, numHid, isSaveModels = True, name = "/home/cheng/DBN.npy", isSingleDBN = True, **kwargs) :
    """implement DBN fitting
    X :data(np.array)
    label : the label of each sample(np.array, in a row)
    numHid : the node of each hidden layer(list)"""

    H = len(numHid)
    m = list()
    nArg = len(kwargs)

    if H >= 2 :
        # train the first rbm model
        if nArg >= 1 :
            string = "layer" + str(1)
            model_ = rbm.rbm(X, numHid[0], **kwargs[string])
        else :
            model_ = rbm.rbm(X, numHid[0])
        m.append(model_)

        if isSingleDBN :
            for index in range(1, H-1) :
                if nArg >= index :
                    string = "layer" +str(index + 1)
                    model_ = rbm.rbm(m[index-1].top, numHid[index], **kwargs[string])
                else :
                    model_ = rbm.rbm(m[index-1].top, numHid[index])
                m.append(model_)

            # train the last rbm model
            if nArg >= H :
                string = "layer" + str(H)
                model_ = rbmFit.rbmFit(m[H-2].top, numHid[H-1], label, **kwargs[string])
            else :
                model_ = rbmFit.rbmFit(m[H-2].top, numHid[H-1], label)
            m.append(model_)
        else :
            for index in range(1, H) :
                if nArg >= index :
                    string = "layer" +str(index + 1)
                    model_ = rbm.rbm(m[index-1].top, numHid[index], **kwargs[string])
                else :
                    model_ = rbm.rbm(m[index-1].top, numHid[index])
                m.append(model_)
    else :
        # only a single layer
        if isSingleDBN :
            if nArg >= 1 :
                string = "layer" + str(1)
                model_ = rbmFit.rbmFit(X, numHid[0], label, **kwargs[string])
            else :
                model_ = rbmFit.rbmFit(X, numHid[0], label)
            m.append(model_)
        else :
            if nArg >= 1 :
                string = "layer" + str(1)
                model_ = rbm.rbm(X, numHid[0], **kwargs[string])
            else :
                model_ = rbm.rbm(X, numHid[0])
            m.append(model_)

    if isSaveModels :
        models = np.array(m)
        np.save(name, models)

    return m

if __name__ == "__main__" :
    data = np.load(sys.argv[1])
    label = np.load(sys.argv[2])
    nHid = [100, 200]

    # layer1 parameters
    p1 = {"maxEpoch" : 100, "modelType" : "BB"}

    # layer2 parameters
    p2 = {"maxEpoch" : 100}

    p = {"layer1" : p1, "layer2" : p2}

    models = DBNFit(data, label, nHid, isSingleDBN=True, **p)
    print len(models)
    print models[0].type," ", models[1].type