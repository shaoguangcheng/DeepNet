import sys
import numpy as np
import os.path as path

if __name__ == "__main__" :
	argv = sys.argv
	argc = len(argv);
	if argc != 3 :
		print "PARAMETER ERROR"
		print "usage : %s %s %s" % (argv[0], "viewBasedData", "shapeBasedData")
		sys.exit(-1)

	viewBasedData = np.load(argv[1])
	shapeBasedData = np.load(argv[2])
	jointData = np.append(viewBasedData, shapeBasedData, axis = 1)

	[baseName, fileName] = path.split(argv[1])
	dataName = path.join(baseName, "viewShapeData.npy")
	np.save(dataName, jointData)
	
