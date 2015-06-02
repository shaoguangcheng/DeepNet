#! /usr/bin/env python

import shelve as sh
import sys

def readShelve(filename) :
	data = sh.open(filename)
	print data

if __name__ == "__main__" :
	argc = len(sys.argv)
	if argc != 2 :
		print "Usage : %s",sys.argv[1], "filename"
		sys.exit(-1)
		
	readShelve(sys.argv[1])
