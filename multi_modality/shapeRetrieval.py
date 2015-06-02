#/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cmath

import sys
import shelve

import multiModalityDBNPredict
import DBNPredict
import rbmPredict

class shapeEvaluation(object) :
    """this class is using to calculate the evaluation index"""

    def __init__(self, simM, label, nQuery = [20, 40, 60, 80]) :
        """simM : the similarity matrix of all shapes
        label : the label of each shape and must satisfy np.array([...]).shape = (n,1)"""
        self.recall = self.calRecall(simM, label, nQuery = nQuery)
        self.precision = self.calPrecision(simM, label, nQuery=nQuery)
        self.NN = self.calNN(simM, label)
        self.FT = self.calFT(simM, label)
        self.ST = self.calST(simM, label)
        self.E  = self.calE(simM, label)
        self.DCG = self.calDCG(simM, label)
        self.RP = self.calRP(simM, label)

    def calRecall(self, simM, label, nQuery = [20, 40, 60, 80]):
        print "calculate recall ..."

        nCase = simM.shape[0]
        nn = len(nQuery)
        recall = {}

        for i in range(nn) :
            nQueryTemp = nQuery[i]
            percent = 0
            for j in range(nCase) :
                labelJ = label[j, 0]
                nLabelJ = np.where(label == labelJ)[0].size

                sortM = np.sort(simM[j, :])
                index = [np.where(k == simM[j, :])[0].tolist()[0] for k in sortM ]
		
                n = len([k for k in label[index[:nQueryTemp], 0] if k == labelJ])

                percent = percent + n*1.0/nLabelJ

            recall[str(i+1)] = percent*1.0/nCase

        return recall

    def calPrecision(self, simM, label, nQuery = [20, 40, 60, 80]):
        print "calculate precision ..."

        nCase = simM.shape[0]
        nn = len(nQuery)
        precision = {}

        for i in range(nn) :
            nQueryTemp = nQuery[i]
            percent = 0
            for j in range(nCase) :
                labelJ = label[j, 0]

                sortM = np.sort(simM[j, :])
                index = [np.where(k == simM[j, :])[0].tolist()[0] for k in sortM]
                n = len([k for k in label[index[:nQueryTemp], 0] if k == labelJ])

                percent = percent + n*1.0/nQueryTemp

            precision[str(i+1)] = percent*1.0/nCase

        return precision

    def calNN(self, simM, label):
        print "calculate NN ..."
        nCase = simM.shape[0]

        percent = 0
        for i in range(nCase) :
            labelI = label[i, 0]

            sortM = np.sort(simM[i, :])
            index = [np.where(k == simM[i, :])[0].tolist()[0] for k in sortM]
            n = len([k for k in label[index[:2], 0] if k == labelI])

            percent = percent + n-1

        NN = percent*1.0/nCase

        return  NN

    def calFT(self, simM, label):
        print "calculate FT ..."

        nCase = simM.shape[0]

        percent = 0
        for i in range(nCase) :
            labelI = label[i, 0]
            nLabelI = np.where(label == labelI)[0].size
            Nt = (nLabelI - 1)

            sortM = np.sort(simM[i, :])
            index = [np.where(k == simM[i, :])[0].tolist()[0] for k in sortM]
            n = len([k for k in label[index[:Nt], 0] if k == labelI])

            if Nt != 0 :
                percent = percent + (n-1)*1.0/Nt

        FT = percent*1.0/nCase

        return FT

    def calST(self, simM, label):
        print "calculate ST ..."

        nCase = simM.shape[0]

        percent = 0
        for i in range(nCase) :
            labelI = label[i, 0]
            nLabelI = np.where(label == labelI)[0].size
            Nt = 2*(nLabelI - 1)

            sortM = np.sort(simM[i, :])
            index = [np.where(k == simM[i, :])[0].tolist()[0] for k in sortM]
            n = len([k for k in label[index[:Nt], 0] if k == labelI])

            if Nt != 0 :
                percent = percent + (n-1)*1.0/Nt

        ST = percent*1.0/nCase

        return ST

    def calE(self, simM, label):
        print "calculate E ..."

        nCase = simM.shape[0]

        percentP = 0
        percentR = 0
        nQuery = 32

        for i in range(nCase) :
            labelI = label[i, 0]
            nLabelI = np.where(label == labelI)[0].size

            sortM = np.sort(simM[i, :])
            index = [np.where(k == simM[i, :])[0].tolist()[0] for k in sortM]
            n = len([k for k in label[index[:nQuery], 0] if k == labelI])

            percentP = percentP + (n-1)*1.0/nQuery
            percentR = percentR + (n-1)*1.0/nLabelI

        percentR = percentR*1.0/nCase
        percentP = percentP*1.0/nCase

        E = 2.0/(1.0/percentP + 1.0/percentR)

        return E

    def calDCG(self, simM, label):
        print "calculate DCG ..."

        nCase = simM.shape[0]

        DCG = 0
        for i in range(nCase) :
            labelI = label[i, 0]
            nLabeI = np.where(label == labelI)[0].size

            sortM = np.sort(simM[i, :])
            index = [np.where(k == simM[i, :])[0].tolist()[0] for k in sortM]
            G = []
            [G.append(int(k == labelI)) for k in label[index,0]]
            G = [G[j+1]*1.0/cmath.log(j+2, 2) for j, k in enumerate(G[1:])]
            DCGI = reduce(lambda x,y :x+y, G, 1)

            temp = [1.0/cmath.log(k,2) for k in range(2, nLabeI+1)]
            DCGZ = reduce(lambda  x,y : x+y, temp, 1)

            DCG = DCG + DCGI*1.0/DCGZ

        DCG = DCG*1.0/nCase

        return DCG.real

    def calRP(self, simM, label) :
        print "calculate RP ..."

        nCase = simM.shape[0]
        nClass  = np.max(label)
        minNumShape = nCase + 1
        for index in range(1, int(nClass)+1) :
            n = np.where(label == index)[0].size
            minNumShape = min(minNumShape, n)

        RP = np.zeros((minNumShape, 2))
        for index in range(1, minNumShape+1) :
            RP[index-1, 0] = index*1.0/minNumShape

        for x in range(nCase) :
            labelX = label[x, 0]
            lastN = 0

            sys.stdout.write(".")
            sys.stdout.flush()

            for y in range(1, nCase+1) :
                sortM = np.sort(simM[x, :])
                idx = [np.where(k == simM[x, :])[0].tolist()[0] for k in sortM]
                n = len([k for k in label[idx[:y], 0] if k == labelX])

                if n - lastN == 1 :
                    RP[n-1,1] = RP[n-1,1] + n*1.0/y
                    lastN = n

                if n >= minNumShape :
                    break

        for index in range(minNumShape) :
            RP[index, 1] = RP[index, 1]*1.0/nCase

        return RP

    def drawPR(self, RP_ = None, isSaveFig = False, figName = "data/RP.eps") :
        "plot Recall-Precision curve"
        print "draw RP ..."

        if RP_ != None :
            RP = RP_
        elif self.RP != None :
            RP = self.RP
        else :
            print "ERROR : no RP data to plot"
            sys.exit()

        plt.figure(figsize=(8,6))
        plt.plot(RP[:, 0], RP[:, 1], label="$PR$", color = "red", linewidth=1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Note : if show() method is executed, then the figure can not be saved.
        if not isSaveFig :
            plt.show()

        if isSaveFig :
            plt.savefig(figName, dpi=200)

    def saveEval(self, isSaveEval = True, fileName = "data/evaluation.shelve"):
        "save computed result"
        print "save evaluation result ...."
        if isSaveEval :
            evaluation = shelve.open(fileName, )
            evaluation["recall"] = self.recall
            evaluation["precision"] = self.precision
            evaluation["NN"] = self.NN
            evaluation["FT"] = self.FT
            evaluation["ST"] = self.ST
            evaluation["E"] = self.E
            evaluation["DCG"] = self.DCG
            evaluation["RP"] = self.RP
            evaluation.close()

def calSimMatrix(costM) :
    "calculate similarity matrix in accordance with cost matrix costM"
    # normalize cost matrix
    maxCostM = np.max(costM, axis=1, keepdims=True)
    minCostM = np.min(costM, axis=1, keepdims=True)
    costM = (costM - minCostM)*1.0/(maxCostM - minCostM)

    nCase = costM.shape[0]
    simM = np.zeros((nCase, nCase))
    for x in range(nCase) :
        simM[x, x] = 0

    for x in range(nCase) :
        for y in range(x+1, nCase) :
            distance = LA.norm(costM[x, :] - costM[y, :])
            simM[x, y] = distance
            simM[y, x] = distance

    maxSimM = np.max(simM)
    minSimM = np.min(simM)

    if np.abs(maxSimM - minSimM) > 1e-5 :
        simM = (simM - minSimM)/(maxSimM - minSimM)
    else :
        print "error in calSimMatrix"
        sys.exit()

    return simM

def testCalRecall() :
    simM = np.array([[0.2, 0.3, 0.1, 0.2], [0.3, 0.4, 0.2, 0.6], \
                     [0.5, 0.2, 0.6, 0.1], [0.1, 0.2, 0.3, 0.4]])
    label = np.array([1,1,2,2]).reshape(4, 1)
    nQuery = [1,2,3]

    evalShape = shapeEvaluation(simM, label, nQuery)
    evalShape.drawPR(isSaveFig=True)

def testEvaluation(modelType="multi-modality", modelsFile = None, viewBasedDataFile = None,
                   shapeBasedDataFile = None, labelFile = None,\
                   isSaveFig = True, figName = None, evaluationFile = None) :
    """modelsFile : multi-modality model file name
    viewBasedDataFile :
    shapeBasedDataFile :
    labelFile :
    isSaveFig : whether save the recall-precision curve or not
    figName : the recall-precision name, only works when isSaveFig is True
    evaluationFile : where to save the retrieval result """

    if modelsFile == None :
        modelsFile = "data/model/multi-modalityDBN.npy"
    if viewBasedDataFile == None :
        viewBasedDataFile = "data/shrec2007/SHREC_2007_BOW_1000_viewBased.npy"
    if shapeBasedDataFile == None :
        shapeBasedDataFile = "data/shrec2007/SHREC_2007_BOW_100_shapeBased.npy"
    if labelFile == None :
        labelFile = "data/shrec2007/label.npy"
    if figName == None :
        figName = "data/shrec2007/RP.eps"
    if evaluationFile == None :
        evaluationFile = "data/shrec2007/evaluation.shelve"

    models = np.load(modelsFile)
    viewBasedData = np.load(viewBasedDataFile)
    shapeBasedData = np.load(shapeBasedDataFile)
    label = np.load(labelFile)

    if modelType == "multi-modality" :
        [y, F] = multiModalityDBNPredict.multiModalityDBNPredict(models, [viewBasedData, shapeBasedData])

        simM = calSimMatrix(F)

        nQuery = [20, 40, 60, 80]
        evalShape = shapeEvaluation(simM, label, nQuery = nQuery)
        evalShape.drawPR(isSaveFig=isSaveFig, figName = figName)
        evalShape.saveEval(fileName=evaluationFile)
    elif modelType == "DBN_view" :
        [yView, FView] = DBNPredict.DBNPredict(models, viewBasedData)

        # process view
        simMView = calSimMatrix(FView)

        nQuery = [20, 40, 60, 80]
        evalShapeView = shapeEvaluation(simMView, label, nQuery = nQuery)

#        figName = figName+".DBN.view"
#        evaluationFile = evaluationFile+".DBN.view"
        evalShapeView.drawPR(isSaveFig=isSaveFig, figName = figName)
        evalShapeView.saveEval(fileName=evaluationFile)

    elif modelType == "DBN_shape" :
        [yShape, FShape] = DBNPredict.DBNPredict(models, shapeBasedData)
        # process shape
        simMShape = calSimMatrix(FShape)

        nQuery = [20, 40, 60, 80]
        evalShapeShape = shapeEvaluation(simMShape, label, nQuery = nQuery)

#        figName = figName+".DBN.shape"
#        evaluationFile = evaluationFile+".DBN.shape"
        evalShapeShape.drawPR(isSaveFig=isSaveFig, figName = figName)
        evalShapeShape.saveEval(fileName=evaluationFile)

    elif modelType == "rbm_view" :
        [yView, FView] = rbmPredict.rbmPredict(models[0], viewBasedData)

        # process view
        simMView = calSimMatrix(FView)

        nQuery = [20, 40, 60, 80]
        evalShapeView = shapeEvaluation(simMView, label, nQuery = nQuery)

#        figName = figName+".rbm.view"
#        evaluationFile = evaluationFile+".rbm.view"
        evalShapeView.drawPR(isSaveFig=isSaveFig, figName = figName)
        evalShapeView.saveEval(fileName=evaluationFile)

    elif modelType == "rbm_shape" :
        [yShape, FShape] = rbmPredict.rbmPredict(models[0], shapeBasedData)
        # process shape
        simMShape = calSimMatrix(FShape)

        nQuery = [20, 40, 60, 80]
        evalShapeShape = shapeEvaluation(simMShape, label, nQuery = nQuery)

#        figName = figName+".rbm.shape"
#        evaluationFile = evaluationFile+".rbm.shape"
        evalShapeShape.drawPR(isSaveFig=isSaveFig, figName = figName)
        evalShapeShape.saveEval(fileName=evaluationFile)


if __name__ == "__main__" :
    testEvaluation()

