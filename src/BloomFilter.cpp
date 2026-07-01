#include "../include/BloomFilter.h"
//#include "../utils/Hash.h"
#include "../utils/xxhash.h"
#include <functional>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;
// to run g++ main.cpp src/BloomFilter.cpp utils/xxhash.c -Iutils -Iinclude -o main
BloomFilter::BloomFilter(size_t size, size_t num_hashes) : m(size),k(num_hashes),bits((size+63)/64,0) {}
BloomFilter::BloomFilter(size_t n, double targetFPR){
    const double ln2 = log(2);
    double m_real = -(n*log(targetFPR))/(ln2*ln2);
    double k_real = (m_real/n)*ln2;
    k = ceil(k_real);
    m = ceil(m_real);
    if(k==0) k = 1;
    bits.resize((m+63)/64,0);
}

void BloomFilter::setBit(size_t idx){
    bits[idx/64] |= (1ULL << (idx%64));
}

bool BloomFilter::getBit(size_t idx) const{
    return bits[idx/64] & (1ULL<<(idx%64));
}

void BloomFilter::insert(const string &key){
    uint64_t h1 = XXH64(key.c_str(), key.length(), 0);
    uint64_t h2 = XXH64(key.c_str(), key.length(), 42);
    if(h2==0) h2 = 1;
    for(size_t i = 0;i<k;i++){
        size_t idx = (h1 + i * h2) % m;
        setBit(idx);
    }   
}

bool BloomFilter::contains(const string &key) const{
    uint64_t h1 = XXH64(key.c_str(), key.length(), 0);
    uint64_t h2 = XXH64(key.c_str(), key.length(), 42);
    if(h2==0) h2 = 1; // if h2 becomes 0 then for all k hashes it would be the same 
    for(size_t i = 0;i<k;i++){
        size_t idx = (h1 + i * h2) % m;
        if(!getBit(idx)) return false;
    }
    return true;
}

void BloomFilter::insert_batch(const vector<string> &keys){
    size_t n = keys.size();
    for(size_t i = 0;i<n;i++){
        insert(keys[i]);
    }
}

vector<uint8_t> BloomFilter::query_batch(const vector<string> &keys) const{
    size_t n = keys.size();
    vector<uint8_t> results(n);
    for(size_t i = 0;i<n;i++){
        results[i] = contains(keys[i]);
    }
    return results;
}

void BloomFilter::clear(){
    fill(bits.begin(),bits.end(),0);
}

size_t BloomFilter::getTheoreticalBits() const{
    return m;
}

size_t BloomFilter::getMemoryBits() const{
    return bits.size()*64;
}

size_t BloomFilter::getMemoryBytes() const{
    return bits.size()*sizeof(uint64_t);
}

double BloomFilter::getBitsPerElement(size_t n) const{
    if(n==0) return 0.0;
    return (double) getMemoryBits()/n;
}

BFStats BloomFilter::benchmark(const vector<string>&testSet,size_t n) const{
    size_t test_n = testSet.size();
    // size_t warm = min((size_t)1000,test_n);
    // for(size_t i = 0;i<warm;i++){
    //     contains(testSet[i]);
    // }
    //auto start = high_resolution_clock::now();
    
    for(size_t i = 0;i<test_n;i++){
        contains(testSet[i]);
    }
    //auto end = high_resolution_clock::now();
    //auto duration_ns = duration_cast<nanoseconds>(end-start).count();

    //double avg_latency = (double)duration_ns/test_n;
    //double throughput = test_n/(duration_ns/1e9);

    BFStats stats;
    // stats.memory_bits = getMemoryBits();
    // stats.bits_per_element = getBitsPerElement(n);
    // stats.avg_latency_ns = avg_latency;
    // stats.throughput_qps = throughput;

    return stats;
}