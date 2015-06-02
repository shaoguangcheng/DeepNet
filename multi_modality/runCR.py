#!/usr/bin/env python

import sys
import os
from optparse import OptionParser as OP
import datetime
import copy

import multiModalityDemo
import DBNdemo
import rbmDemo
import shapeRetrieval

def shapeCR(opts) :
    "do shape classification and retrieval"
    if "-a" in sys.argv :
        sys.argv.remove("-a")
    if "--action" in sys.argv :
        sys.argv.remove("--action")

    midfix = str(datetime.datetime.now()).split(".")[0]
    path = os.path.split(opts.label)[0]

    path = os.path.join(path, midfix+"_"+opts.action)
    os.mkdir(path)

    if opts.action == "multi-modality" :
        # do classification
        [options, args] = multiModalityDemo.parseOptions(sys.argv)

        # if we want to save the classification result, we set its name here
        if options.isSaveResult :
            options.resultName = path+"/"+midfix+"_multi_modality_classification.shelve"

        # set the path to save the trained model
        options.model = path+"/"+midfix+"_multi_modality_model.npy"

        multiModalityDemo.testMultiModalityDBN(options)

        # do retrieval
        # R-P curve name
        figName = path+"/"+midfix+"_multi_modality_RP.eps"

        # retrieval result file name
        evaluationFile = path+"/"+midfix+"_multi_modality_retrieval.shelve"
        shapeRetrieval.testEvaluation(modelType="multi-modality", modelsFile=options.model,
                                      viewBasedDataFile=opts.viewBasedFeature,
                                      shapeBasedDataFile=opts.shapeBasedFeature,
                                      labelFile=opts.label,
                                      isSaveFig=True,
                                      figName=figName,
                                      evaluationFile=evaluationFile)

    elif opts.action == "DBN_view" :
        if "--viewBasedFeature" in sys.argv :
            sys.argv.remove("--viewBasedFeature")
        elif "-v" in sys.argv :
            sys.argv.remove("-v")

        sys.argv.append("--feature")
        sys.argv.append(str(opts.viewBasedFeature))

        # do classification using view Based feature

        (options, args) = DBNdemo.parseOptions(sys.argv)

        # if we want to save the classification result, we set its name here
        if options.isSaveResult :
            options.resultName = path+"/"+midfix+"_DBN_view_classification.shelve"

        # set the path to save the trained view based model
        options.model = path+"/"+midfix+"_DBN_view_model.npy"
        DBNdemo.testDBN(options)

        # do retrieval using trained model
        # R-P curve name
        figName = path+"/"+midfix+"_DBN_view_RP.eps"

        # retrieval result file name
        evaluationFile = path+"/"+midfix+"_DBN_view_retrieval.shelve"
        shapeRetrieval.testEvaluation(modelType="DBN_view", modelsFile=options.model,
                                      viewBasedDataFile=opts.viewBasedFeature,
                                      shapeBasedDataFile=opts.shapeBasedFeature,
                                      labelFile=opts.label,
                                      isSaveFig=True,
                                      figName=figName,
                                      evaluationFile=evaluationFile)

    elif opts.action == "DBN_shape" :
        if "--shapeBasedFeature" in sys.argv :
            sys.argv.remove("--shapeBasedFeature")
        elif "-s" in sys.argv :
            sys.argv.remove("-s")

        sys.argv.append("--feature")
        sys.argv.append(str(opts.shapeBasedFeature))

        # do classification using view Based feature
        [options, args] = DBNdemo.parseOptions(sys.argv)

        # if we want to save the classification result, we set its name here
        if options.isSaveResult :
            options.resultName = path+"/"+midfix+"_DBN_shape_classification.shelve"

        # set the path to save the trained shape based model
        options.model = path+"/"+midfix+"_DBN_shape_model.npy"
        DBNdemo.testDBN(options)

        # do retrieval using trained model
        # R-P curve name
        figName = path+"/"+midfix+"_DBN_shape_RP.eps"

        # retrieval result file name
        evaluationFile = path+"/"+midfix+"_DBN_shape_retrieval.shelve"
        shapeRetrieval.testEvaluation(modelType="DBN_shape", modelsFile=options.model,
                                      viewBasedDataFile=opts.viewBasedFeature,
                                      shapeBasedDataFile=opts.shapeBasedFeature,
                                      labelFile=opts.label,
                                      isSaveFig=True,
                                      figName=figName,
                                      evaluationFile=evaluationFile)
    elif opts.action == "rbm_view" :
        if "--viewBasedFeature" in sys.argv :
            sys.argv.remove("--viewBasedFeature")
        elif "-v" in sys.argv :
            sys.argv.remove("-v")

        sys.argv.append("--feature")
        sys.argv.append(str(opts.viewBasedFeature))

        # do classification using view Based feature
        (options, args) = rbmDemo.parseOptions(sys.argv)

        # if we want to save the classification result, we set its name here
        if options.isSaveResult :
            options.resultName = path+"/"+midfix+"_rbm_view_classification.shelve"

        # set the path to save the trained shape based model
        options.model = path+"/"+midfix+"_rbm_view_model.npy"
        rbmDemo.testRBM(options)

        # do retrieval using trained model
        # R-P curve name
        figName = path+"/"+midfix+"_rbm_view_RP.eps"

        # retrieval result file name
        evaluationFile = path+"/"+midfix+"_rbm_view_retrieval.shelve"
        shapeRetrieval.testEvaluation(modelType="rbm_view", modelsFile=options.model,
                                      viewBasedDataFile=opts.viewBasedFeature,
                                      shapeBasedDataFile=opts.shapeBasedFeature,
                                      labelFile=opts.label,
                                      isSaveFig=True,
                                      figName=figName,
                                      evaluationFile=evaluationFile)
    elif opts.action == "rbm_shape" :
        if "--shapeBasedFeature" in sys.argv :
            sys.argv.remove("--shapeBasedFeature")
        elif "-s" in sys.argv :
            sys.argv.remove("-s")

        sys.argv.append("--feature")
        sys.argv.append(str(opts.shapeBasedFeature))

        # do classification using view Based feature
        (options, args) = rbmDemo.parseOptions(sys.argv)

        # if we want to save the classification result, we set its name here
        if options.isSaveResult :
            options.resultName = path+"/"+midfix+"_rbm_shape_classification.shelve"

        # set the path to save the trained shape based model
        options.model = path+"/"+midfix+"_rbm_shape_model.npy"
        rbmDemo.testRBM(options)

        # do retrieval using trained model
        # R-P curve name
        figName = path+"/"+midfix+"_rbm_shape_RP.eps"

        # retrieval result file name
        evaluationFile = path+"/"+midfix+"_rbm_shape_retrieval.shelve"
        shapeRetrieval.testEvaluation(modelType="rbm_shape", modelsFile=options.model,
                                      viewBasedDataFile=opts.viewBasedFeature,
                                      shapeBasedDataFile=opts.shapeBasedFeature,
                                      labelFile=opts.label,
                                      isSaveFig=True,
                                      figName=figName,
                                      evaluationFile=evaluationFile)

def parseOptions(sysArgv) :
    "parse options from command line"
    # remove the redundant parameter for this file
    options = ['-a', '-v', '-s', '-l', "--action", "--viewBasedFeature", "--shapeBasedFeature", "--label"]
    argv = copy.deepcopy(sysArgv)
    for x in argv :
        if x.startswith('-') and x not in options :
            argv.remove(x)

    parser = OP(usage="usage : %prog --action program")
    parser.add_option("-a", "--action",
                      action="store",
                      dest="action",
                      default="multi-modality",
                      help="using which method to do classification and retrieval. there are three possible options : (1) rbm  (2) DBN (3) multi-modality\n")
    parser.add_option("-v", "--viewBasedFeature",
                      action="store",
                      dest="viewBasedFeature",
#                      default='data/shrec2007/SHREC_2007_BOW_1000_viewBased.npy',
                      help="Feature file name")
    parser.add_option("-s", "--shapeBasedFeature",
                      action="store",
                      dest="shapeBasedFeature",
#                      default='data/shrec2007/SHREC_2007_BOW_100_shapeBased.npy',
                      help="Feature file name")
    parser.add_option("-l", "--label",
                      action="store",
                      dest="label",
#                      default='data/shrec2007/label.npy',
                      help="Label file name")
#    parser.add_option("-m", "--model",
#                      action="store",
#                      dest="model",
#                      default='data/model/multiModalityModel.npy',
#                      help="multi modality model file name to save")
    (opts, args) = parser.parse_args(argv)

    print("---------------------------------------------------------------")
    print("opts = %s" % str(opts))
    print("args = %s" % str(args))
    print("---------------------------------------------------------------")

    return  (opts, args)
if __name__ == "__main__" :
    (opts, args) = parseOptions(sys.argv)
    shapeCR(opts)
