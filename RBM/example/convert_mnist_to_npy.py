#!/usr/bin/env python


import os,sys
import numpy as np


def convert_txt2npy(fname, opts):
    """Convert txt data file to numpy .npy file"""
    
    # get output file name
    fa = os.path.splitext(fname)

    if( len(fa) < 2 ):
        print("ERR: input file name not correct! %s\n" % fname)
        return
        
    fn_out = fa[0] + '.npy'

    print('process %s -> %s' % (fname, fn_out))

    # load txt file
    a = np.loadtxt(fname, dtype=opts.dtype)
    
    # fix label array
    if( len(a.shape) == 1 ):
        a = a.reshape(a.shape[0], 1)

    # save to .npy
    np.save(fn_out, a)


def parse_arguments():
    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options] filenames",
                          version="%prog 1.0")

    parser.add_option("-s", "--sailent",
                      action="store_true",
                      dest="sailent",
                      default=False,
                      help="Do not show informations")
    parser.add_option("-d", "--dtype",
                      action="store",
                      dest="dtype",
                      default="float",
                      help="npy data type (default float)")


    (opts, args) = parser.parse_args()

    if( len(args) < 1 ):
        parser.error("Wrong number of arguments, please input file name!")

    if( not opts.sailent ):
        print("---------------------------------------------------------------")
        print("opts = %s" % str(opts))
        print("args = %s" % str(args))
        print("---------------------------------------------------------------")
        print("")

    return (opts, args)


if( __name__ == '__main__' ):
    # parse input arguments
    (opts, args) = parse_arguments()

    # process each file
    for f in args:
        convert_txt2npy(f, opts)
