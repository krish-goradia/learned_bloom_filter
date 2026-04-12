#ifndef HASH_UTIL_H
#define HASH_UTIL_H

#include<string>
#include<functional>
using namespace std;

class HashUtil{
public:
    static size_t hash1(const string &key){
        return std::hash<std::string>{}(key);
    }
    static size_t hash2(const string &key){
        return std::hash<std::string>{}(key+"salt");
    }
};


#endif