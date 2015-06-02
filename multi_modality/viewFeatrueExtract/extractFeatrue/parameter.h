#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
class parameter
{
public:
    enum LLCMethod{defaultMethod = 0,SUM,MAX};

    //declaration
    static double lambda;
    static int knn;
    static LLCMethod method;
    static string imageFilePath;
    static int nVocabulary;
    static string pathToSaveBOW;

    void print()
    {
        cout << "lambda        : " << lambda << endl;
        cout << "knn           : " << knn << endl;
        cout << "LLCMethod     : " << method<< endl;
        cout << "imageFilePath : " << imageFilePath << endl;
        cout << "nVocabulary   : " << nVocabulary << endl;
        cout << "pathToSaveBOW : " << pathToSaveBOW << endl;
        cout << endl;
    }

    void saveParameters()
    {
        FILE *fp;
        if((fp = fopen("parameters--record.txt","a+")) == NULL){
            printf("can not open file : %s\n","parameters--record.txt");
            return;
        }

        time_t tmVal;
        struct tm *tmPtr;
        char buf[256];

        time(&tmVal);
        tmPtr = localtime(&tmVal);
        sprintf(buf,"%d:%d:%d, %d-%d-%d",tmPtr->tm_hour,tmPtr->tm_min,
                tmPtr->tm_sec,tmPtr->tm_year+1900,tmPtr->tm_mon+1,tmPtr->tm_mday);

        fprintf(fp,"---%s---\n",buf);
        fprintf(fp,"imageFilePath : %s\n",imageFilePath.c_str());
        fprintf(fp,"nVocabulary   : %d\n",nVocabulary);
        fprintf(fp,"pathToSaveBOW : %s\n",pathToSaveBOW.c_str());
        fprintf(fp,"method        : %d\n",method);
        fprintf(fp,"lambda        : %lf\n",lambda);
        fprintf(fp,"knn           : %d\n",knn);
        fprintf(fp,"\n");

        fclose(fp);
    }
};

//default parameters(definition and initialization)
parameter::LLCMethod parameter::method    = parameter::defaultMethod;
double parameter::lambda    = 1e-4;
int parameter::knn       = 5;
int parameter::nVocabulary = -1;
string parameter::imageFilePath = "";
string parameter::pathToSaveBOW = "";

#endif // PARAMETER_H
