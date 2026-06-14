#include "../include/Vectorizer.h"
#include "../utils/xxhash.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#if defined(_MSC_VER)
    #include <xmmintrin.h>
    #define PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T1)
#else
    #define PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#endif
using namespace std;
using namespace chrono;

Vectorizer::Vectorizer(int n_features, int min_n, int max_n)
    : n_features(n_features), min_n(min_n), max_n(max_n) {}

double Vectorizer::sigmoid(double x) const {
    return 1.0 / (1.0 + exp(-x));
}

// Extract char ngrams via rolling hash, hash to feature index, aggregate counts
// Rolling hash: O(1) per ngram vs O(n) for xxHash on substring
// Only supports single ngram size (min_n == max_n)
vector<pair<int,float>> Vectorizer::extract_features(const string& url) const {
    vector<float> counts(n_features, 0.0f);
    vector<int> dirty;
    dirty.reserve(128);

    const int      ng       = min_n;
    const int      len      = (int)url.size();
    const uint64_t BASE     = 131;
    const uint64_t MASK     = (uint64_t)(n_features - 1);

    if (len < ng) return {};

    uint64_t base_pow = 1;
    for (int i = 0; i < ng - 1; i++)
        base_pow *= BASE;

    uint64_t h = 0;
    for (int i = 0; i < ng; i++)
        h = h * BASE + (unsigned char)url[i];

    auto add = [&](uint64_t hash) {
        int idx = (int)(hash & MASK);
        if (counts[idx] == 0.0f) dirty.push_back(idx);
        counts[idx] += 1.0f;
    };

    add(h);
    for (int i = ng; i < len; i++) {
        h -= (unsigned char)url[i - ng] * base_pow;
        h  = h * BASE + (unsigned char)url[i];
        add(h);
    }

    vector<pair<int,float>> features;
    features.reserve(dirty.size());
    for (int idx : dirty)
        features.emplace_back(idx, counts[idx]);
    sort(features.begin(), features.end());

    for (int idx : dirty) counts[idx] = 0.0f;
    return features;
}


CSRMatrix Vectorizer::transform(const vector<string>& urls) const {
    CSRMatrix csr;
    csr.n_rows = (int)urls.size();
    csr.n_cols = n_features;
    csr.indptr.reserve(urls.size() + 1);
    csr.indptr.push_back(0);

    for (const auto& url : urls) {
        auto features = extract_features(url);
        for (auto& [idx, val] : features) {
            csr.indices.push_back(idx);
            csr.data.push_back(val);
        }
        csr.indptr.push_back((int)csr.indices.size());
    }

    return csr;
}

ScoredResult Vectorizer::transform_and_score(
    const vector<string>& urls,
    const vector<double>& weights,
    double intercept) const
{
    size_t n = urls.size();

    vector<float> counts(n_features, 0.0f);
    vector<int>   dirty;
    dirty.reserve(128);

    const int      ng       = min_n;
    const uint64_t BASE     = 131;  // larger prime — better distribution
    const uint64_t MASK     = (uint64_t)(n_features - 1);  // bitmask, n_features power of 2

    // BASE^(ng-1) computed once, no MOD
    uint64_t base_pow = 1;
    for (int i = 0; i < ng - 1; i++)
        base_pow *= BASE;

    auto score_one = [&](const string& url) -> double {
        dirty.clear();
        const int len = (int)url.size();

        if (len >= ng) {
            // first window — no MOD
            uint64_t h = 0;
            for (int i = 0; i < ng; i++)
                h = h * BASE + (unsigned char)url[i];

            auto add = [&](uint64_t hash) {
                int idx = (int)(hash & MASK);
                if (counts[idx] == 0.0f) dirty.push_back(idx);
                counts[idx] += 1.0f;
            };

            add(h);

            // slide — natural uint64 wraparound, no MOD
            for (int i = ng; i < len; i++) {
                h -= (unsigned char)url[i - ng] * base_pow;
                h  = h * BASE + (unsigned char)url[i];
                add(h);
            }
        }

        // dot product over dirty indices only
        double logit = intercept;
        for (int idx : dirty)
            logit += weights[idx] * counts[idx];

        // reset dirty only
        for (int idx : dirty) counts[idx] = 0.0f;

        // fast sigmoid — no exp()
        return 1.0 / (1.0 + exp(-logit));
    };

    // warmup
    size_t warm = min((size_t)1000, n);
    for (size_t i = 0; i < warm; i++) score_one(urls[i]);

    // timed run with prefetch
    vector<double> probs(n);
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < n; i++) {
        if (i + 1 < n)
            PREFETCH(urls[i + 1].c_str());
        probs[i] = score_one(urls[i]);
    }
    auto end = high_resolution_clock::now();

    double total_ns = (double)duration_cast<nanoseconds>(end - start).count();

    ScoredResult result;
    result.probs          = probs;
    result.avg_latency_ns = total_ns / n;
    result.throughput_qps = n / (total_ns / 1e9);
    return result;
}