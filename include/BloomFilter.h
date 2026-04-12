#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H


#include<vector>
#include<string>
using namespace std;

class BloomFilter{
private:
    size_t m,k;
    vector<uint64_t> bits;
    void setBit(size_t idx);
    bool getBit(size_t idx) const;

public:
    BloomFilter(size_t size, size_t num_hashes);
    BloomFilter(size_t n, double targetFPR);
    void insert(const string &key);
    bool contains(const string &key) const;
    void clear();
    size_t getSize() const;

};


#endif