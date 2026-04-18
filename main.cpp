#include <iostream>
#include "include/BloomFilter.h"
#include <vector>
#include <unordered_set>
#include <random>
#include <chrono>


using namespace std;
using namespace chrono;

int main(){
    size_t n = 1e6;
    double targetFPR = 0.07;
    
    BloomFilter bf(n,targetFPR);
    vector<uint64_t> insertSet;
    vector<uint64_t> testSet;
    unordered_set<uint64_t> insertedSet;

    mt19937_64 rng(42);
    for(size_t i  = 0;i<n;i++){
        uint64_t val = rng();
        insertSet.push_back(val);
        insertedSet.insert(val);
    }

    for(size_t i = 0;i<n;i++){
        uint64_t val = rng();
        if(insertedSet.find(val)==insertedSet.end()) testSet.push_back(val);
    }

    for(const auto &x: insertSet){
        bf.insert(x);
    }

    size_t falsePositiveCount = 0;
    auto start = high_resolution_clock::now();
    for(const auto&x:testSet){
        if(bf.contains(x)) falsePositiveCount++;
    }
    auto end = high_resolution_clock::now();


    double fpr = (double)falsePositiveCount/testSet.size();
    double duration = duration_cast<nanoseconds>(end-start).count();
    double avgQueryTime = duration/testSet.size();
    double bitsPerElement = double(bf.getSize())/n;
    double sizeofBf = bf.getSize() /(8*1024.0 * 1024.0);
    

    cout << "n = " << n << endl;
    cout << "FPR = " << fpr << endl;
    cout << "bits/element = " << bitsPerElement << endl;
    cout << "size (MB) of bf = " << sizeofBf << endl;
    cout << "avg query time (nanoseconds) = " << avgQueryTime << endl;
    return 0;

}