#/usr/bin python

"""using this when process shape feature"""

import os
import numpy as np
import sys


fileList = sys.argv[1]
nShape = int(sys.argv[2])
nBow = int(sys.argv[3])

sep = ""

# convert txt data format to npy format
def collect() :
	fid = open(fileList,"r")
	shapePrefix = fid.readlines()
	
	nDim = int((1+nBow)*nBow*1.0/2)
	feature = np.zeros((nShape, nDim))
	for index, line in enumerate(shapePrefix) :
		line = line.strip('\r\n') + "_ssbof.dat"
		print line
		temp = np.fromfile(line, dtype=np.float64, count=-1)
		temp = temp[1:]

		k = 0
		temp = temp.reshape(nBow,nBow)
		for i in range(nBow) :
			for j in range(i,nBow) :
				feature[index,k] = temp[i,j]
				k = k + 1

	(base, name) = os.path.split(fileList)
	pos = name.rfind('_')
	name = name[ : pos]+"_shapeBased_BOW.npy"
	featureName = os.path.join(base, name)
	np.save(featureName, feature)	
		

if __name__ == "__main__" :
    collect()
