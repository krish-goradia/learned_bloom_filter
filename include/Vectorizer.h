#ifndef VECTORIZER_H
#define VECTORIZER_H

#include <vector>
#include <string>
#include <cstdint>

struct CSRMatrix {
    std::vector<float>  data;     // all non-zero values (counts)
    std::vector<int>    indices;  // column indices
    std::vector<int>    indptr;   // row pointers
    int                 n_rows;
    int                 n_cols;   // = n_features
};

struct ScoredResult {
    std::vector<double> probs;
    double avg_latency_ns;
    double throughput_qps;
};

class Vectorizer {
public:
    Vectorizer(int n_features, int min_n, int max_n);

    CSRMatrix transform(const std::vector<std::string>& urls) const;

    ScoredResult transform_and_score(
        const std::vector<std::string>& urls,
        const std::vector<double>& weights,
        double intercept
    ) const;

private:
    int n_features;
    int min_n;
    int max_n;

    std::vector<std::pair<int,float>> extract_features(const std::string& url) const;
    double sigmoid(double x) const;
};

#endif