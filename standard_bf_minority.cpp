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
    size_t bad_cnt = 0;
    double targetFPR = 0.07;
    
    ifstream badfile("bad_urls.txt");
    string url;
    while(getline(badfile,url)) bad_cnt++;
    badfile.close();

    BloomFilter standardBF(bad_cnt==0?1:bad_cnt,targetFPR);

    ifstream badurls("bad_urls.txt");
    while(getline(badurls,url)) standardBF.insert(url);
    badurls.close();

    ifstream fu("test_urls.txt");
    ifstream fl("test_labels.txt");

    size_t totalNeg = 0,falsePos = 0;
    string label_str;
    int label;
    while(getline(fu,url) &&getline(fl, label_str)){
        bool result;
        label = stoi(label_str);
        result = standardBF.contains(url);
        if(label==1){
            totalNeg++;
            if(result) falsePos++;
        }
        
    }
    double fpr = totalNeg? (double) falsePos/totalNeg:0.0;

    cout << "FPR=" << fpr << endl;
    cout << "Size=" << standardBF.getSize()/(8*1024.0 * 1024.0) << endl;

    return 0;
}