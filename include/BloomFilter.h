#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H


#include<vector>
#include<string>
using namespace std;

struct BFStats{
    size_t memory_bits;
    double bits_per_element;
    double avg_latency_ns;
    double throughput_qps;
};


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
    void insert_batch(const vector<string> &keys);
    bool contains(const string &key) const;
    vector<uint8_t> query_batch(const vector<string> &keys) const;
    void clear();
    size_t getTheoreticalBits() const;
    size_t getMemoryBytes() const;
    size_t getMemoryBits() const;
    double getBitsPerElement(size_t n) const;
    BFStats benchmark(const vector<string>&testSet,size_t n) const;

};


#endif