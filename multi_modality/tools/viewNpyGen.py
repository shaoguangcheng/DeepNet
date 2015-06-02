#!/usr/bin python

"""convert feature data with txt format to npy format, including feature data labels
   use this when process view feature"""

import numpy as np
import os
import sys

dataPath = sys.argv[1]
(prefix, featureData) = os.path.split(dataPath)
nShape = int(sys.argv[2])

sep = " "

def convert():
        featureFileName = os.path.join(dataPath)
        feature = np.fromfile(featureFileName, dtype=np.float64, count=-1, sep=sep)
        feature = feature.reshape((nShape, -1))

        label = feature[:, [0]]
        feature = feature[:, 2:]

        featureFileName = os.path.join(prefix, featureData.split('.')[0]+".npy")
        labelFileName = os.path.join(prefix, featureData.split('.')[0]+"_label.npy")
        np.save(featureFileName, feature)
        np.save(labelFileName, label)


if __name__ == "__main__" :
    convert()
