#include <iostream>
#include "include/BloomFilter.h"
#include <vector>
#include <unordered_set>
#include <random>
#include <chrono>
#include <fstream>


using namespace std;
using namespace chrono;

int main(){
    size_t fn_count = 0 ;
    double targetFPR = 0.01;

    ifstream countFile("fn.txt");
    string temp;
    while(getline(countFile,temp)) fn_count++;
    countFile.close();

    BloomFilter backupBF(fn_count==0?1:fn_count,targetFPR);

    ifstream fnfile("fn.txt");
    while(getline(fnfile,temp)) backupBF.insert(temp);
    fnfile.close();

    ifstream fu("test_urls.txt");
    ifstream fl("test_labels.txt");
    ifstream fp("test_probs.txt");

    double threshold = 0.9;
    size_t totalNeg = 0,falsePos =0,totalPos = 0;
    string url;
    int label;
    double prob;

    while(getline(fu,url) && (fl>>label) && (fp>>prob)){
        bool result;
        if(prob>=threshold) result = true;
        else result = backupBF.contains(url);

        if(label==0){
            totalNeg++;
            if(result)falsePos++;
        }
        else totalPos++;
    }

    double fpr = totalNeg? (double) falsePos/totalNeg :0.0;
    cout << "FN_count=" << fn_count << endl;
    cout << "FPR=" << fpr << endl;
    cout << "Size=" << backupBF.getSize() /(8*1024.0 * 1024.0) << endl;
    cout << "totalNeg=" << totalNeg << endl;
    cout << "totalPos=" << totalPos << endl;

    return 0;

}