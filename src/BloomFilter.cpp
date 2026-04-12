#include "../include/BloomFilter.h"
//#include "../utils/Hash.h"
#include "../utils/xxhash.h"
#include <functional>
#include <algorithm>
#include <cmath>

using namespace std;
// to run g++ main.cpp src/BloomFilter.cpp utils/xxhash.c -Iutils -Iinclude -o main
BloomFilter::BloomFilter(size_t size, size_t num_hashes) : m(size),k(num_hashes),bits((size+63)/64,0) {}
BloomFilter::BloomFilter(size_t n, double targetFPR){
    m = -(n*log(targetFPR))/(log(2)*log(2));
    k = (m/n)*log(2);
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

void BloomFilter::clear(){
    fill(bits.begin(),bits.end(),0);
}

size_t BloomFilter::getSize() const{
    return m;
}