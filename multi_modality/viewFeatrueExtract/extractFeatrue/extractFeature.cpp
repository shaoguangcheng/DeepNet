#include "extractFeature.h"
#include "parameter.h"

#include <iostream>

using namespace std;

void help()
{
    cout << "======================================================="<< endl;
    cout << "[usage] : extractFeature depthImagePath nVoacabulary prefixOfSaveBOW knn llcMethod lambda" << endl;
    cout << "=======================================================" << endl;
}

int main(int argc,char* argv[])
{

    if(argc < 4){
        help();
        return -1;
    }

    uint64 start = getTickCount();

    parameter::imageFilePath = string(argv[1]);
    parameter::nVocabulary = atoi(argv[2]);
    parameter::pathToSaveBOW = string(argv[3]);

    if(argc >= 5)
        parameter::knn = atoi(argv[4]);
    if(argc >= 6)
        parameter::method = parameter::LLCMethod(atoi(argv[5]));
    if(argc >= 7)
        parameter::lambda =atof(argv[6]);

    parameter p;
    p.saveParameters();
    p.print();

    vector<SiftDescriptor> siftDescriptorForAll,siftDescriptorAfterMerge;
    extractSiftDescriptor(parameter::imageFilePath,siftDescriptorForAll);
    mergeModelSiftDescriptor(siftDescriptorForAll,siftDescriptorAfterMerge);

    vector<BOWDescriptor> bow;
    computeBOW(siftDescriptorAfterMerge,parameter::nVocabulary,bow, parameter::pathToSaveBOW, true);
  //  computeLLCBOW(siftDescriptorAfterMerge,parameter::nVocabulary,bow, parameter::pathToSaveBOW, true);

  //  vector<TF_IDF> tf_idf;
  //  calTF_IDF(bow,tf_idf,parameter::pathToSaveBOW);

  //  vector<ZCA> zca;
  //  cal_ZCA(bow,zca,parameter::pathToSaveBOW,true);

    uint64 finish = getTickCount();

    cout << "===================================" << endl;
    cout << "depth images path    : " << parameter::imageFilePath << endl;
    cout << "total models         : " << bow.size() << endl;
    cout << "number of vocabulary : " << parameter::nVocabulary << endl;
    cout << "time consumed        : " << (finish-start)/getTickFrequency() << endl;
    cout << "===================================" << endl;

    return 0;

//	Mat m;
//	m.create(3000,128,5);
//	loadMat(string(argv[1]), m);
//	cout << m << endl;
}
