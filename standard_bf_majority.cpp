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
    size_t good_cnt = 0;
    double targetFPR = 0.05;

    ifstream goodfile("good_urls.txt");
    string url;
    while(getline(goodfile,url)) good_cnt++;
    goodfile.close();

    BloomFilter standardBF(good_cnt==0?1:good_cnt,targetFPR);

    ifstream goodurls("good_urls.txt");
    while(getline(goodurls,url)) standardBF.insert(url);
    goodurls.close();

    ifstream fu("test_urls.txt");
    ifstream fl("test_labels.txt");

    size_t totalNeg = 0,falsePos = 0;
    size_t totalqueries = 0;
    int label;
    auto start = high_resolution_clock::now();
    while(getline(fu,url) &&(fl>>label)){
        totalqueries++;
        bool result;
        if(label==0){
            totalNeg++;
            result = standardBF.contains(url);
            if(result) falsePos++;
        }
    }
    auto end = high_resolution_clock::now();

    double fpr = totalNeg? (double) falsePos/totalNeg:0.0;
    double duration = duration_cast<nanoseconds>(end-start).count();
    double sizeofBf = standardBF.getSize()/(8*1024.0 * 1024.0);
    double avgQueryTime = duration/totalqueries;
    double bitsPerElement = double(standardBF.getSize())/good_cnt;
    double sizeofBf = standardBF.getSize()/(8*1024.0 * 1024.0);

    cout << "FPR=" << fpr << endl;
    cout << "Size=" << sizeofBf << endl;
    cout << "bits/element=" << bitsPerElement << endl;
    cout << "avg_query_time_(nanoseconds)=" << avgQueryTime << endl;
   

    return 0;
}