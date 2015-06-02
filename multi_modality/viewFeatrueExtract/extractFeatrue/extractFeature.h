#ifndef EXTRACTSIFT_H
#define EXTRACTSIFT_H

#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Eigen/SVD"
#include "Eigen/Dense"

#include "parameter.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <iterator>

#define randChoose 1

using namespace cv;
using namespace std;
using namespace Eigen;

struct featureInfo{
    int catergory;
    int num;
    int angle;
    int view;
};

struct featureDescriptor{
    featureInfo featureInformation;
    Mat featureDescriptorData;
};

typedef featureDescriptor SiftDescriptor;
typedef featureDescriptor BOWDescriptor;
typedef featureDescriptor TF_IDF;
typedef featureDescriptor ZCA;


void extractSiftDescriptor(string &imageFileName, Mat &siftDescriptorForSingleImage)
{
    Mat image = imread(imageFileName,CV_LOAD_IMAGE_GRAYSCALE);
//    Mat outImage;
    vector<KeyPoint> siftKeyPoint;
    SIFT sift;

    sift.detect(image,siftKeyPoint);
    sift.compute(image,siftKeyPoint,siftDescriptorForSingleImage);

//    drawKeypoints(image,siftKeyPoint,outImage);
//    imshow("sift key points",outImage);
//    waitKey();
}

void extractSiftDescriptor(string imageFilePath,vector<SiftDescriptor> &siftDescriptorForAll,
                           string siftPrefix = "/home/cheng",bool saveSift = false)
{
    string imageFileName;

    ifstream inFile;
    inFile.open(imageFilePath.c_str());
    if(!inFile.is_open()){
        cout << "can not open " << imageFilePath << endl;
        return;
    }

    int lineNum;
    Mat siftDescriptorForSingleImage;
    SiftDescriptor siftDescriptor;
    while(!inFile.eof()){
        inFile >> lineNum;
        inFile >> siftDescriptor.featureInformation.catergory;
        inFile >> siftDescriptor.featureInformation.num;
        inFile >> imageFileName;

        extractSiftDescriptor(imageFileName,siftDescriptorForSingleImage);
        cout << "view " << (lineNum-1)%200 + 1 << " of "
             << "model " << siftDescriptor.featureInformation.num << " in catergory "
             << siftDescriptor.featureInformation.catergory << " : " << siftDescriptorForSingleImage.rows
             << " sift points" << endl;

        siftDescriptor.featureDescriptorData = siftDescriptorForSingleImage;
        siftDescriptorForAll.push_back(siftDescriptor);
    }

    inFile.close();

    siftDescriptorForAll.resize(siftDescriptorForAll.size()-1);//because of eof, remove the last data term

    //check sift
    if(saveSift){
        ofstream outFile;
        string siftName = siftPrefix + "_siftFeature.txt";
        outFile.open(siftName.c_str());
        if(!outFile.is_open()){
            cout << "can not create " << siftName << endl;
            return;
        }

        vector<SiftDescriptor>::iterator it = siftDescriptorForAll.begin();
        for(;it != siftDescriptorForAll.end();it++){
            outFile << it->featureInformation.catergory << " "
                    << it->featureInformation.num << endl;
    //                << it->siftInformation.angle << " "
    //                << it->siftInformation.view << " ";

            outFile << it->featureDescriptorData << endl;
        }

        outFile.close();
    }

}

void adjacentMat(Mat &dest,const Mat &src)
{
    assert(dest.cols == src.cols);
    dest.push_back(src);
}


void mergeModelSiftDescriptor(vector<SiftDescriptor> &siftDescriptorForAll,
                              vector<SiftDescriptor> &siftDescriptorAfterMerge,
                              bool saveMergeModelSiftDescriptor = false)
{
    vector<SiftDescriptor>::iterator currentViewSift = siftDescriptorForAll.begin();
    vector<SiftDescriptor>::iterator nextViewSift = siftDescriptorForAll.begin()+1;
    for(;currentViewSift != siftDescriptorForAll.end()&&
        nextViewSift != siftDescriptorForAll.end();
        currentViewSift++,nextViewSift++){
        SiftDescriptor       modelSift;
        modelSift.featureInformation.catergory = currentViewSift->featureInformation.catergory;
        modelSift.featureInformation.num       = currentViewSift->featureInformation.num ;
        modelSift.featureDescriptorData        = currentViewSift->featureDescriptorData;

        while(nextViewSift->featureInformation.catergory == currentViewSift->featureInformation.catergory&&
                nextViewSift->featureInformation.num == currentViewSift->featureInformation.num&&
              nextViewSift != siftDescriptorForAll.end()){
            adjacentMat(modelSift.featureDescriptorData,nextViewSift->featureDescriptorData);

            nextViewSift++;
            currentViewSift++;
        }

        siftDescriptorAfterMerge.push_back(modelSift);
    }

    if(saveMergeModelSiftDescriptor){
        ofstream outFile;

        outFile.open("featureData/MergeModelSiftDescriptor.txt");
        if(!outFile.is_open()){
            cout << "can not open featureData/MergeModelSiftDescriptor.txt" << endl;
            return;
        }

        vector<SiftDescriptor>::iterator it = siftDescriptorAfterMerge.begin();
        for(;it != siftDescriptorAfterMerge.end();it++){
            outFile << it->featureInformation.catergory << " "
                    << it->featureInformation.num << " ";

            outFile << it->featureDescriptorData << endl;
        }

        outFile.close();
    }
}

void chooseModelToCluster(vector<pair<int,int> > &modelToClusterPerCategory,int nModelPerCategory,int categoryLabel,double nModelRatioToChoose)
{
    vector<int> randNumber;

    for(int i=0;i<nModelPerCategory;i++){
        randNumber.push_back(i);
    }

    random_shuffle(randNumber.begin(),randNumber.end());

//    ostream_iterator<int> print(cout," ");
//    copy(randNumber.begin(),randNumber.end(),print);
//    cout << endl;

    for(int i=0;i<int(nModelPerCategory*nModelRatioToChoose);i++){
        modelToClusterPerCategory.push_back(make_pair(categoryLabel,randNumber[i]));//modelToClusterPerCategory.push_back();
    }
}

void randomChooseModelToGenerateBOW(vector<SiftDescriptor> &siftDescriptorAfterMerge, vector<vector<pair<int,int> > > &modelToCluster)
{
    int categoryLabel = 1,nModelPerCategory = 0;

    vector<SiftDescriptor>::iterator itModel = siftDescriptorAfterMerge.begin();
    for(;itModel != siftDescriptorAfterMerge.end();itModel++){
        if(itModel->featureInformation.catergory == categoryLabel){
            nModelPerCategory++;
        }
        else{
            vector<pair<int,int> > modelToClusterPerCategory;

            chooseModelToCluster(modelToClusterPerCategory,nModelPerCategory,categoryLabel,0.7);
            modelToCluster.push_back(modelToClusterPerCategory);

            categoryLabel = itModel->featureInformation.catergory;
            nModelPerCategory = 1;
        }
    }

    vector<pair<int,int> > modelToClusterPerCategory;
    chooseModelToCluster(modelToClusterPerCategory,nModelPerCategory,categoryLabel,0.7);
    modelToCluster.push_back(modelToClusterPerCategory);

    cout << "model to generate words:" << endl;
    for(int i=0;i<int(modelToCluster.size());i++){
        for(int j=0;j<int(modelToCluster[i].size());j++){
            cout << "category : " << modelToCluster[i][j].first << " num : " << modelToCluster[i][j].second << endl;
        }
    }
}

/*
 *hint : something to show to make other body know what the computer is doing now
 */
void saveFeature(vector<featureDescriptor> &bowDescriptor,const string &fileName,const string &hint = "")
{
    cout << hint << endl;
    ofstream outFile;

    outFile.open(fileName.c_str());
    if(!outFile.is_open()){
        cout << "can not open " << fileName << endl;
        return;
    }

    vector<featureDescriptor>::iterator BOWIt = bowDescriptor.begin();
    for(;BOWIt != bowDescriptor.end();BOWIt++){
        outFile << BOWIt->featureInformation.catergory << " "
                << BOWIt->featureInformation.num << " ";

        outFile << BOWIt->featureDescriptorData << endl;
    }

    outFile.close();
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//these functions can be reused
void saveMat(const Mat &matrix,const string &fileName)
{
    ofstream outFile;

    outFile.open(fileName.c_str());
    if(!outFile.is_open()){
        cout << "can not open file : " << fileName << endl;
        return;
    }

    outFile << matrix <<endl;
    outFile.close();
}

//load Mat format matrix
void loadMat(const string &fileName,Mat &matrix)
{
    assert(matrix.data != NULL);
    ifstream inFile;

    inFile.open(fileName.c_str());
    if(!inFile.is_open()){
        cout << "can not open file : " << fileName << endl;
        return;
    }

    int i = 0;
    float *ptr = (float*)matrix.data;
    char temp;
    inFile >> temp;
    while(!inFile.eof()){
        char num[4];
        inFile >> num;

        ptr[i++] = (float)atof(num);
        if(i == matrix.cols*matrix.rows)
            break;
    }
}

//compute trace
template<class T>
T tr(const Mat_<T> &m)
{
    assert(m.cols == m.rows);

    T t;
    T* ptr = (T*)m.data;
    for(int i=0;i<m.rows;i++)
        t += ptr[i*m.rows+i];
    return t;
}

void bubleSort(double* data,int* index,int length)
{
    assert(data != NULL&&index != NULL&&length > 0);
    for(int i=0;i<length-1;i++)
        for(int j=i;j<length;j++)
            if(data[i]>data[j]){
                double d = data[i];
                data[i] = data[j];
                data[j] = d;

                int t = index[i];
                index[i] = index[j];
                index[j] = t;
            }

}

template <class T>
inline void printElement(const T &container,const char* delim = "")
{
    typename T::const_iterator pos;
    for(pos = container.begin();pos != container.end();pos++){
        cout << *pos << delim;
    }
    cout << endl;
}

template <class T>
void loadDatArray(const string &fileName,uint64_t &rows,uint64_t &cols,T **data)
{
    int i;
    T *data_;
    FILE *fp = fopen(fileName.c_str(),"r");
    if(!fp){
        printf("can not open file : %s",fileName.c_str());
        return;
    }

    i = fread(&rows,sizeof(uint64_t),1,fp);
    i = fread(&cols,sizeof(uint64_t),1,fp);

    data_ = new T [rows*cols];
    i = fread(data_,sizeof(T)*rows*cols,1,fp);

    fclose(fp);

    *data = data_;
}

inline double log2(double x)
{
    assert(x>0);
    return log(x)/log(2);
}

void maxDimsion(Mat_<double> src,Mat_<double> &dst,Mat_<int> &index)
{
    assert(src.cols == dst.cols&&src.cols == index.cols);
    double maxNumber;
    double *ptrSrc = (double*)src.data;
    double *ptrDst = (double*)dst.data;
    int    *ptrIndex = (int*)index.data;
    int index_ = 0;
    for(int i=0;i<src.cols;i++){
        maxNumber = -999.0;
        for(int j=0;j<src.rows;j++){
            if(maxNumber < ptrSrc[i+j*src.cols]){
                maxNumber = ptrSrc[i+j*src.cols];
                index_ = j;
            }
        }

        ptrDst[i] = maxNumber;
        ptrIndex[i]  = index_;
    }
    return;
}

////////////////////////////////////////////////////////
//compute PCA whiten and ZCA whiten using opencv libray
////////////////////////////////////////////////////////

//calculate mean of matrix x by column
template <class T>
void calMean(const Mat_<T> &x,Mat_<T> &avg)
{
    assert(x.cols == avg.cols);
    assert(avg.data != NULL&&x.data != NULL);

    T* xPtr = (T*)x.data;
    T* avgPtr  = (T*)avg.data;

    for(int i=0;i<x.rows;i++)
        for(int j=0;j<x.cols;j++)
            avgPtr[j] += xPtr[i*x.cols+j];

    for(int j=0;j<x.cols;j++)
        avgPtr[j] /= x.rows;
}

template <class T>
void cal_ZCA_PCA(const Mat_<T> x,Mat_<T> &zca,Mat_<T> &pca)
{
    Mat_<T> avg(1,x.cols,0.0);
    calMean(x,avg);

    T* xPtr = (T*)x.data;
    T* avgPtr  = (T*)avg.data;
    for(int i=0;i<x.rows;i++)
        for(int j=0;j<x.cols;j++)
            xPtr[i*x.cols+j] -= avgPtr[j];

    Mat_<T> sigma(x.cols,x.cols,0.0);
    sigma = x.t()*x/x.rows;

    Mat_<T> U(x.cols,x.cols,0.0),
            S(1,x.cols,0.0),
            Vt(x.cols,x.cols,0.0);
    SVD::compute(sigma,S,U,Vt,SVD::MODIFY_A);

    Mat_<T> xRot(x.cols,x.rows,0.0);

    xRot = U.t()*x.t();

    pca = pca.t();
    zca = zca.t();
    float elipson = 1e-4;
    T *xRotPtr = (T*)xRot.data;
    T *SPtr = (T*)S.data;
    T *pcaPtr = (T*)pca.data;
    for(int i=0;i<x.cols;i++)
        for(int j=0;j<x.rows;j++){
            pcaPtr[i*x.rows+j] = xRotPtr[i*x.rows+j]/(sqrt(SPtr[i]+elipson));
        }

    zca = U * pca;

    pca = pca.t();
    zca = zca.t();
}

////////////////////////////////////////////////
///compute PCA whiten and ZCA whiten using eigen3
/////////////////////////////////////////////////

void ZCANormalize(MatrixXf &zca)
{
    assert(zca.data() != NULL);

    VectorXf minPerRow(zca.rows()),maxPerRow(zca.rows());
    for(int i=0;i<zca.rows();i++){
        minPerRow(i) = 999.0;
        maxPerRow(i) = -999.0;
    }

    //find maximun and minimun element in each row
    for(int i=0;i<zca.rows();i++)
        for(int j=0;j<zca.cols();j++){
            if(zca(i,j) < minPerRow(i))
                minPerRow(i) = zca(i,j);
            if(zca(i,j) > maxPerRow(i))
                maxPerRow(i) = zca(i,j);
        }

    //normalize each element to [0,1]
    for(int i=0;i<zca.rows();i++)
        for(int j=0;j<zca.cols();j++)
            zca(i,j) = (zca(i,j)-minPerRow(i))/(maxPerRow(i)-minPerRow(i));

    return;
}

void calMean(const MatrixXf &x,VectorXf &avg)
{
    assert(x.cols() == avg.rows());
    assert(x.data() != NULL && avg.data() != NULL);

    for(int i=0;i<x.rows();i++)
        for(int j=0;j<x.cols();j++)
            avg(j) += x(i,j);

    avg /= x.rows();
}

void cal_ZCA_PCA(MatrixXf &x,MatrixXf &zca,MatrixXf &pca)
{
    VectorXf avg(x.cols());
    avg.setZero(x.cols(),1);

    calMean(x,avg);
    for(int i=0;i<x.rows();i++)
        for(int j=0;j<x.cols();j++)
            x(i,j) -= avg(j);

    MatrixXf sigma(x.cols(),x.cols());
    sigma.setZero(x.cols(),x.cols());
    sigma = x.transpose()*x/x.rows();

    JacobiSVD<MatrixXf> svd(sigma,ComputeThinU|ComputeThinV);
    VectorXf singularValue(x.cols());
    singularValue = svd.singularValues();

    MatrixXf xRot(x.cols(),x.rows());
    xRot = svd.matrixU().transpose() * x.transpose();

    pca = pca.transpose().eval();//this equal to pca.transposeInPlace();
                                 //notice : "pca = pca.transpose()" is wrong.
    zca.transposeInPlace();

    float elipson = 1e-4;
    for(int i=0;i<x.cols();i++)
        for(int j=0;j<x.rows();j++)
            pca(i,j) = xRot(i,j)/(sqrt(singularValue(i)+elipson));

    zca = svd.matrixU()*pca;

    pca = pca.transpose().eval();
    zca.transposeInPlace();

    ZCANormalize(zca);

    return;
}

//convert Mat(opencv) to MatrixX(Eigen) : has been tested
template <class T>
Matrix<T,Dynamic,Dynamic> Mat2MatrixX(const Mat_<T> &P)
{
    Matrix<T,Dynamic,Dynamic> Q(P.rows,P.cols);
    Q.setZero(P.rows,P.cols);

    T *PPtr = (T*)P.data;
    for(int i=0;i<P.rows;i++)
        for(int j=0;j<P.cols;j++)
            Q(i,j) = PPtr[i*P.cols+j];

    return Q;
}

//convert MatrixX(Eigen) to Mat(opencv) : has been tested
template <class T>
Mat_<T> MatrixX2Mat(const Matrix<T,Dynamic,Dynamic> &Q)
{
    Mat_<T> P(Q.rows(),Q.cols());
    T *PPtr = (T*)P.data;
    for(int i=0;i<Q.rows();i++)
        for(int j=0;j<Q.cols();j++)
            PPtr[i*Q.cols()+j] = Q(i,j);

    return P;
}
/////////////////////////////////////////////////////////////////////////////////

//locality constrained linear coding(LLC)
int computeLLC( Mat_<double> vocabulary, Mat_<double> feature,
               Mat_<double> &sparseCode,int* knnID = NULL)
{
    int knn = parameter::knn;
    double lambda = parameter::lambda;

    assert(vocabulary.cols == feature.cols);
    assert(feature.rows == 1);
    assert(vocabulary.rows >= knn);

    int nVocabulary = vocabulary.rows;
    int featureDimsion = feature.cols;

    int *dIndex = new int[nVocabulary];
    double *da = new double[nVocabulary];
    double distance;

    double *ptrVocabulary = (double*)vocabulary.data;
    double *ptrFeature    = (double*)feature.data;
    for(int i=0;i<nVocabulary;i++){
        distance = 0.0;
        for(int j=0;j<featureDimsion;j++)
            distance += (ptrVocabulary[i*featureDimsion+j] - ptrFeature[j])*
                    (ptrVocabulary[i*featureDimsion+j] - ptrFeature[j]);

        da[i] = distance;
        dIndex[i] = i;
    }

   // quickSort();
    bubleSort(da,dIndex,nVocabulary);
    if(knnID != NULL) *knnID = dIndex[0];

    Mat_<double> z(knn,featureDimsion),C(knn,knn);
    Mat1d b(knn,1,1),w(knn,1,0.0);

    double*ptrZ = (double*)z.data;
    for(int i=0;i<knn;i++){
        for(int j=0;j<featureDimsion;j++){
            ptrZ[i*featureDimsion+j] = ptrVocabulary[dIndex[i]*featureDimsion+j]-ptrFeature[j];
        }
    }

    Mat_<double> zTranspose(featureDimsion,knn);
    transpose(z,zTranspose);

    C = z * zTranspose;
    double* ptrC = (double*)C.data;
    for(int i=0;i<knn;i++){
        ptrC[i*knn+i] += lambda*tr(C);
    }

    solve(C,b,w,DECOMP_QR);

    distance = 0;
    double* ptrW = (double*)w.data;
    for(int i=0;i<knn;i++){
        distance += ptrW[i];
    }

    for(int i=0;i<knn;i++){
        ptrW[i] /= distance;
        ptrW[i] =(ptrW[i]+1.0)/2;
    }

    double* ptrS = (double*)sparseCode.data;
    for(int i=0;i<knn;i++){
        ptrS[dIndex[i]] = ptrW[i];
    }

    delete dIndex;
    delete da;
    return 0;
}

int computeLLCBOWForSingleModel(const SiftDescriptor &siftForSingleModel,const Mat_<double> vocabulary,
                                BOWDescriptor &LLCBOWForSingleModel)
{
    LLCBOWForSingleModel.featureInformation.catergory = siftForSingleModel.featureInformation.catergory;
    LLCBOWForSingleModel.featureInformation.num       = siftForSingleModel.featureInformation.num;

    int method = parameter::method;

    //sum method
    if(method == 1){
        Mat_<double> BOWForSingleModel(1,vocabulary.rows,0.0);
        for(int i=0;i<siftForSingleModel.featureDescriptorData.rows;i++){
            Mat_<double> feature(siftForSingleModel.featureDescriptorData.row(i),0.0);
            Mat_<double> sparseCode(1,vocabulary.rows,0.0);

            computeLLC(vocabulary,feature,sparseCode);
            BOWForSingleModel += sparseCode;
        }

        LLCBOWForSingleModel.featureDescriptorData = BOWForSingleModel;
        return 0;
    }

    //max method
    if(method == 2){
        Mat_<double> LLCSingleModel(siftForSingleModel.featureDescriptorData.rows,vocabulary.rows,0.0);
        for(int i=0;i<siftForSingleModel.featureDescriptorData.rows;i++){
            Mat_<double> feature(siftForSingleModel.featureDescriptorData.row(i));
            Mat_<double> sparseCode(1,vocabulary.rows,0.0);

            computeLLC(vocabulary,feature,sparseCode);
            LLCSingleModel.push_back(sparseCode);
        }

        Mat_<double> BOWForSingleModel(1,vocabulary.rows,0.0);
        Mat_<int> index(1,vocabulary.rows,0);
        maxDimsion(LLCSingleModel,BOWForSingleModel,index);

        LLCBOWForSingleModel.featureDescriptorData = BOWForSingleModel;
        return 0;
//////////////////////////////to be cintinued here
    }

    return 0;
}

//compute bow with LLC
void computeLLCBOW(vector<SiftDescriptor> &siftDescriptorAfterMerge,
                   int nVocabulary,vector<BOWDescriptor> &bowDescriptor,
                    string BOWFilePrefix, bool saveBOW = true)
{
    assert(siftDescriptorAfterMerge.size() > 1);

#if randChoose
    vector<vector<pair<int,int> > > modelToCluster;
    randomChooseModelToGenerateBOW(siftDescriptorAfterMerge,modelToCluster);
#endif

    BOWKMeansTrainer BOWTraining(nVocabulary);
    Mat vocabulary;

    vector<SiftDescriptor>::iterator it = siftDescriptorAfterMerge.begin();
    for(;it != siftDescriptorAfterMerge.end();it++){

#if randChoose
        int cat = 0,num = 0;
        for(;cat < int(modelToCluster.size());cat++)
            for(;num < int(modelToCluster[cat].size());num++)
                 if(it->featureInformation.catergory == modelToCluster[cat][num].first&&
                    it->featureInformation.num == modelToCluster[cat][num].second)
#endif
                        BOWTraining.add(it->featureDescriptorData);

        }

    cout << "   clustering ... " << endl;
    vocabulary = BOWTraining.cluster();

    string vocabularyName = BOWFilePrefix + "_LLCvocabulary.txt";
    saveMat(vocabulary,vocabularyName);

//    Mat_<double> vocabulary(nVocabulary,128,0.0);
//    loadMat(vocabularyName,vocabulary);

    it = siftDescriptorAfterMerge.begin();
    for(;it != siftDescriptorAfterMerge.end();it++){
        BOWDescriptor LLCBOWForSingleModel;
        computeLLCBOWForSingleModel(*it,vocabulary,LLCBOWForSingleModel);//choose max method to compute LLC

        bowDescriptor.push_back(LLCBOWForSingleModel);
    }

    if(saveBOW){
        string BOWFileName = BOWFilePrefix + "_LLCBOW.txt";
        saveFeature(bowDescriptor,BOWFileName,"saving LLCBOW ... ");
    }
}


void computeBOW(vector<SiftDescriptor> &siftDescriptorAfterMerge, int nVocabulary,vector<BOWDescriptor> &bowDescriptor,
               string BOWFilePrefix,  bool saveBOW = false)
{
    assert(siftDescriptorAfterMerge.size() > 1);

    Mat vocabulary;
    vector<SiftDescriptor>::iterator it;

    #if randChoose
    vector<vector<pair<int,int> > > modelToCluster;
    randomChooseModelToGenerateBOW(siftDescriptorAfterMerge,modelToCluster);
    #endif

        BOWKMeansTrainer BOWTraining(nVocabulary);

        it = siftDescriptorAfterMerge.begin();
        for(;it != siftDescriptorAfterMerge.end();it++){

    #if randChoose
            int cat = 0,num = 0;
            for(;cat < int(modelToCluster.size());cat++)
                for(;num < int(modelToCluster[cat].size());num++)
                     if(it->featureInformation.catergory == modelToCluster[cat][num].first&&
                        it->featureInformation.num == modelToCluster[cat][num].second)
    #endif
                            BOWTraining.add(it->featureDescriptorData);

            }

        cout << "   clustering ... " << endl;
        vocabulary = BOWTraining.cluster();

        string vocabularyName = BOWFilePrefix + "_vocabulary.txt";
        saveMat(vocabulary,vocabularyName);

        vocabulary.create(nVocabulary,128, CV_32F);
        loadMat(vocabularyName, vocabulary);
   

    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2,false);//ensure every feature can find a cluster center

    for(it = siftDescriptorAfterMerge.begin();it != siftDescriptorAfterMerge.end();it++){
      matcher.match(it->featureDescriptorData,vocabulary,matches);

      Mat BOWDescriptorForSingleModel = Mat(1,nVocabulary,CV_32FC1,Scalar::all(0.0));
      float *ptr = (float*)BOWDescriptorForSingleModel.data;//be careful float,can not be double type,

      for(int i=0;i<int(matches.size());i++){
          int trainIdx = matches[i].trainIdx;
          ptr[trainIdx] = ptr[trainIdx]+1.0;
      }

      BOWDescriptor bowForSingleModel;
      bowForSingleModel.featureInformation.catergory = it->featureInformation.catergory;
      bowForSingleModel.featureInformation.num       = it->featureInformation.num;
      bowForSingleModel.featureDescriptorData        = BOWDescriptorForSingleModel;

      bowDescriptor.push_back(bowForSingleModel);

      cout << "BOW for model " << bowForSingleModel.featureInformation.num << " in category "
           << bowForSingleModel.featureInformation.catergory << " generated" << endl;
    }

    if(saveBOW){
        cout << "   saving BOW ..." << endl;
        ofstream outFile;
		ofstream outOrigFile;

        string BOWFileName = BOWFilePrefix + "_BOW.txt";
        outFile.open(BOWFileName.c_str());
        if(!outFile.is_open()){
            cout << "can not open " << BOWFileName << endl;
            return;
        }

		string origBOWFileName = BOWFilePrefix + "_orig_BOW.txt";
		outOrigFile.open(origBOWFileName.c_str());

        vector<BOWDescriptor>::iterator BOWIt = bowDescriptor.begin();
        for(;BOWIt != bowDescriptor.end();BOWIt++){
            outFile << BOWIt->featureInformation.catergory << " "
                    << BOWIt->featureInformation.num << " ";
			
            outOrigFile << BOWIt->featureInformation.catergory << " "
                    << BOWIt->featureInformation.num << " ";
			outOrigFile << format(BOWIt->featureDescriptorData, "csv") << endl;

            double s=-1;
            for(int i=0;i<BOWIt->featureDescriptorData.cols;i++){
                if(BOWIt->featureDescriptorData.at<float>(i) > s)
                    s = BOWIt->featureDescriptorData.at<float>(i);
            }
		
            BOWIt->featureDescriptorData /= s;
            outFile << format(BOWIt->featureDescriptorData, "csv") << endl;
        }

        outFile.close();
		outOrigFile.close();
    }

}

//ZCA whiten
void cal_ZCA(const vector<BOWDescriptor> &bowDescriptor,vector<ZCA> &zca, string zcaPrefix, bool saveZCA = 1)
{
    const int nModels = bowDescriptor.size();
    const int nVoc   = parameter::nVocabulary;

    Mat_<float> bowData1;
    vector<BOWDescriptor>::const_iterator bowIt = bowDescriptor.begin();
    for(;bowIt != bowDescriptor.end();bowIt++)
        bowData1.push_back(bowIt->featureDescriptorData);

    MatrixXf bowData2 = Mat2MatrixX(bowData1);
    MatrixXf zca_(nModels,nVoc),pca_(nModels,nVoc);
    zca_.setZero(nModels,nVoc);
    pca_.setZero(nModels,nVoc);

    cal_ZCA_PCA(bowData2,zca_,pca_);

    for(;bowIt != bowDescriptor.end();bowIt++)
        bowData1.pop_back();
    bowData1 = MatrixX2Mat(zca_);

    int i = 0;
    bowIt = bowDescriptor.begin();
    for(;bowIt != bowDescriptor.end();bowIt++){
        ZCA zcaForSingleModel;
        zcaForSingleModel.featureInformation.catergory = bowIt->featureInformation.catergory;
        zcaForSingleModel.featureInformation.num = bowIt->featureInformation.num;
        zcaForSingleModel.featureDescriptorData = bowData1.row(i++);

        zca.push_back(zcaForSingleModel);
    }

    if(saveZCA){
        string zcaName = zcaPrefix + "_zca.txt";
        saveFeature(zca,zcaName,"saving ZCA ... ");
    }

    return;
}

//tf-idf : compute weight for each vocabulary in per model
void calTF_IDF(const vector<BOWDescriptor> &bowDescriptor,vector<TF_IDF> &tf_idf, string tf_idf_prefix, bool saveTF_IDF = 1)
{
   const int nModels = bowDescriptor.size();
   const int nVoc   = parameter::nVocabulary;

   int *df = new int[nVoc];
   assert(df != NULL);
   for(int i=0;i<nVoc;i++)
       df[i] = 0;

   //calculate df.
    vector<BOWDescriptor>::const_iterator it;
    for(it = bowDescriptor.begin();it!=bowDescriptor.end();it++){
        float *bowPtr = (float*)(it->featureDescriptorData.data);//here make me fixed
        for(int i=0;i<nVoc;i++)
            if(bowPtr[i] > 0)
                df[i]++;
    }

//    copy(df,df+nVoc,ostream_iterator<int> (cout," "));
    for(it = bowDescriptor.begin();it!=bowDescriptor.end();it++){
        TF_IDF tf_idfForSingleModel;
        Mat_<float> tf_idfData(1,nVoc,0.0);

        tf_idfForSingleModel.featureInformation.num = it-> featureInformation.num;
        tf_idfForSingleModel.featureInformation.catergory = it->featureInformation.catergory;

        float *bowPtr = (float*)(it->featureDescriptorData.data);
        float *tf_idfPtr = (float*)tf_idfData.data;
        for(int i=0;i<nVoc;i++)
            tf_idfPtr[i] = bowPtr[i]*log2((nModels*1.0)/df[i]);

        tf_idfForSingleModel.featureDescriptorData = tf_idfData;
        tf_idf.push_back(tf_idfForSingleModel);
    }

    delete [] df;

    if(saveTF_IDF){
        string tf_idf_name = tf_idf_prefix + "_tf_idf.txt";
        saveFeature(tf_idf,tf_idf_name,"saving TF-IDF ... ");
    }

    return;
}

////////////////////////////////////////////////////////////////
//test LLC
double testLLC()
{
    uint64_t nVocabulary,featureDimision,nFeature;
    double *ptrVocabulary,*ptrFeature,*oCode;
    double err = 0.0;
    int i;

    loadDatArray("test_llc_sc_dic.dat",nVocabulary,featureDimision,&ptrVocabulary);
    loadDatArray("test_llc_sc_X.dat",nFeature,featureDimision,&ptrFeature);
    loadDatArray("test_llc_sc_c.dat",nFeature,nVocabulary,&oCode);

    Mat_<double> vocabulary(nVocabulary,featureDimision,0.0),feature(nFeature,featureDimision,0.0),cCode;

    double *p = (double*)vocabulary.data;
    for(i=0;i<int(nVocabulary*featureDimision);i++){
        p[i] = ptrVocabulary[i];
    }

    p = (double*)feature.data;
    for(i=0;i<int(nFeature*featureDimision);i++){
        p[i] = ptrFeature[i];
    }

 //   memcpy((double*)vocabulary.data,ptrVocabulary,nVocabulary*featureDimision);
 //   memcpy((double*)feature.data,ptrFeature,nFeature*featureDimision);

    for(i=0;i<int(nFeature);i++){
        Mat_<double> code(1,nVocabulary,0.0);
        computeLLC(vocabulary,feature.row(i),code);
        cCode.push_back(code);
    }

    cout << cCode << endl;

    p = (double*)cCode.data;
    for(i=0;i<int(nFeature*nVocabulary);i++)
        err += fabs(p[i]-oCode[i]);

    return err;
}

#endif // EXTRACTSIFT_H
