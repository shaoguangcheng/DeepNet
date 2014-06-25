DeepNet
============

In this project, I implement a deep learning toolbox (DeepNet) including RBM, DBN, Multi-modal DBN with Python， in which the majority of matrix operations are carried on GPU by using the Cudamat to speed up the calculation process. There are some examples to show how to use this package.

This project make some references to the matlab code in https://github.com/dmus/API-Project. However, in comparison with the matlab
code, our version improves the performance 4 times.

Requirements 
============
.NumPy (http://www.numpy.org/)
> 
.Cudamat (already included, https://github.com/cudamat/cudamat)

Usage 
============
To use the toolbox, following steps are needed.
> 
(1) compile the Cudamat library :
  ```
  cd (directory to DeepNet)
  cd DeepNet/RBM/cudamat/
  Make （ note : correct path to gcc-4.6 or below version should be given in Makefile ）
  ```

(2) change directory to RBM/, then set the DEEPNET_PATH variable in set_env.sh file to the RBM/ path in your computer

(3) run command :
  ```source set_env.sh```
  
(4) We provide some demo programs to this toolbox. For RBM and DBN demos, we use Mnist data. To run these demos, you should first 
uncompress the data in example/.
  > 
  ```
  cd example/
  tar -xzvf mnist_data.tar.gz
  python rbmDemo.py
  ```
  or
  > 
  ```python DBNdemo.py```
  
  For any help information, run
  > 
  ```python rbmDemo.py --help```
  > 
  or
  > 
  ```python DBNdemo.py --help```
  
  rbm demo usage ：
  > 
    Usage: rbmDemo.py [options] filenames
  
    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -p TRAINPERCENT, --trainPercent=TRAINPERCENT
                            Trainning data percentage
      -e MAXEPOCH, --maxEpoch=MAXEPOCH
                            Iteration number
      -f FEATURE, --feature=FEATURE
                            Feature file name
      -l LABEL, --label=LABEL
                            Label file name
      -m MODEL, --model=MODEL
                            DBN model file name
      -b ISSAVERESULT, --verbose=ISSAVERESULT
                            whether to save classification result or not
      -n RESULTNAME, --name=RESULTNAME
                            the file name of classification result, only works
                            when -b is true
   
   DBN demo usage :
   > 
     Usage: DBNdemo.py [options] filenames
  
     Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -p TRAINPERCENT, --trainPercent=TRAINPERCENT
                            Trainning data percentage
      -e MAXEPOCH, --maxEpoch=MAXEPOCH
                            Iteration number
      -f FEATURE, --feature=FEATURE
                            Feature file name
      -l LABEL, --label=LABEL
                            Label file name
      -m MODEL, --model=MODEL
                            DBN model file name
      -b ISSAVERESULT, --verbose=ISSAVERESULT
                            whether to save classification result or not
      -n RESULTNAME, --name=RESULTNAME
                            the file name of classification result, only works
                            when -b is true
                            
  For multi-modal demo, we use data in our paper "Multi-modal Feature Fusion for 3D Shape Recognition and Retrieval". 
  To run this demo, change directory to multi-modal_demo/ and run 
  ```python multiModalityDemo.py```
    
  multi-modal demo usage :
  > 
    Usage: multiModalityDemo.py [options] args

    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
      -p TRAINPERCENT, --trainPercent=TRAINPERCENT
                              Trainning data percentage
      -e MAXEPOCH, --maxEpoch=MAXEPOCH
                              Iteration number
      -v VIEWBASEDFEATURE, --viewBasedFeature=VIEWBASEDFEATURE
                              Feature file name
      -s SHAPEBASEDFEATURE, --shapeBasedFeature=SHAPEBASEDFEATURE
                              Feature file name
      -l LABEL, --label=LABEL
                              Label file name
      -m MODEL, --model=MODEL
                              multi modality model file name to save
      -b ISSAVERESULT, --verbose=ISSAVERESULT
                              whether to save classification result or not
      -n RESULTNAME, --name=RESULTNAME
                              the file name of classification result, only works
                              when -b is true
                              
Platform 
===========
This code is only test on Linux mint-16 64-bit.

Contact 
===========
If you have any question about this code, please contact me directly.
E-mail : chengshaoguang1291@gmail.com

        
    


