#!/usr/bin/env python

import shelve

"rbm model can also  be implemented by using dict"
class rbmModel(object) :
    """define rbm model"""

    def __init__(self, weight, biasV, biasH, type = None, top = None, weightLabel = None, biasLabel = None, labels = None):
        "define the parameter of rbm \
        weight : weight between visible layer and hidden layer \
        biasV  : the bias of visible layer \
        biasH  : the bias of hidden layer \
        type   : rbm type \
        top    : the output of hidden layer(label layer does not have this parameter) \
        weightLabel : the weight of label layer(only in last layer of DBN and single layer rbm) \
        biasLabel   : the bias of label layer"

        self.weight = weight
        self.biasV = biasV
        self.biasH = biasH
        self.type = type
        self.top = top
        self.weightLabel = weightLabel
        self.biasLabel = biasLabel
        self.labels = labels

    def save(self, name = "data/rbm.shelve") :
        "save single rbm model using numpy as *.npy format"
        m = shelve.open(name)
        m["model"] = self
        m.close()

    def load(self, name = "data/rbm.shelve") :

        m = shelve.open(name)
        self = m["model"]
        return self

if __name__ == "__main__" :
    pass
