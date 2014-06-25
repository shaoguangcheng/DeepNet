DeepNet
============

In this project, I implement a deep learning toolbox (DeepNet) including RBM, DBN, Multi-modal DBN with Python， in which the majority of matrix operations are carried on GPU by using the Cudamat to speed up the calculation process. There are some examples to show how to use this package.

This project make some references to the matlab code in https://github.com/dmus/API-Project. However, in comparison with the matlab
code, our version improves the performance 4 times.

Requirements 
============
* NumPy (http://www.numpy.org/)
* Cudamat (already included, https://github.com/cudamat/cudamat)

Usage 
============
To use the toolbox, following steps are needed.

(1) compile the Cudamat library :
  ```python
  cd (directory to DeepNet)
  cd DeepNet/RBM/cudamat/
  Make （ note : correct path to gcc-4.6 or below version should be given in Makefile ）
  ```

(2) change directory to RBM/, then set the DEEPNET_PATH variable in set_env.sh file to the RBM/ path in your computer

(3) run command :
  ```
  source set_env.sh
  ```
  
(4) We provide some demo programs in this toolbox.
> 
  **RBM and DBN demos**
  ---------------
  For RBM and DBN demos, we use Mnist data, which has been contained in our toolbox. To run these demos, you should first uncompress the data in example/.
  ```python
  cd example/
  tar -xzvf mnist_data.tar.gz
  python rbmDemo.py
  or
  python DBNdemo.py
  ```
  
  For help information, run
  ```python
  python rbmDemo.py --help
  or
  python DBNdemo.py --help
  ```
  **Multi-modal demo**
  ---------------
  For multi-modal demo, we employ SHREC 2007 data to show the usage.  How the data is generated has been elaborated in paper  "Multi-modal Feature Fusion for 3D Shape Recognition and Retrieval". 
  To run this demo, change directory to multi-modal_demo/ and run 
  ```python
  python multiModalityDemo.py
  ```
    
                              
Platform 
===========
This code is only tested on Linux mint-16 64-bit.

Contact 
===========
If you have any question about this code, please contact me directly.
E-mail : chengshaoguang1291@gmail.com

        
    


