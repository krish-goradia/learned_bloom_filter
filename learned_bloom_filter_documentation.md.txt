Learned Bloom Filters — Project Documentation
Research Supervisor: Prof. Anirban Dasgupta, IIT Gandhinagar
Duration: January 2026 – April 2026
Dataset: 420K URLs (binary classification: good/bad)


________________


1. Motivation
A classical Bloom Filter (BF) is a space-efficient probabilistic data structure that answers membership queries with a tunable false positive rate (FPR) and a strict zero false negative guarantee. However, it treats every element identically — it has no notion that some elements are more likely to be queried than others.


The Learned Bloom Filter (Kraska et al., 2018) replaces part of the BF with a learned model that acts as a pre-filter. The model classifies most elements correctly, and a much smaller backup BF handles the model's mistakes. The hypothesis is that for structured data like URLs, the model can generalize well enough that the backup BF is significantly smaller than a full BF, yielding memory savings at the same FPR target.


________________


2. System Architecture
2.1 Standard Bloom Filter (Baseline)
Query URL → Bloom Filter (stores all bad training URLs) → bad / good


All bad URLs from the training set are inserted. At query time, the BF returns:


* True positive (bad URL found): flag as bad
* False positive (good URL found): flag as bad (bounded by target FPR)
* True negative (good URL not found): flag as good
* False negative: impossible by BF construction
2.2 Learned Bloom Filter
Query URL → ML Model → prob >= threshold?


                YES  → flag as bad


                NO   → query Backup BF → found?


                              YES → flag as bad


                              NO  → flag as good


The backup BF is populated with bad URLs that the model misclassified on the training set (false negatives). Its role is to ensure training-time model errors do not propagate to the output.


________________


3. Implementation
3.1 Bloom Filter (C++17)
Built from scratch using:


* xxHash64 with double hashing: idx = (h1 + i * h2) % m for i in 0..k
* Bit array stored as vector<uint64_t> for cache-friendly access
* Two constructors: direct (m, k) or optimal (n, target_fpr) using the standard formula:
   * m = -(n * ln(fpr)) / ln(2)²
   * k = (m/n) * ln(2)
* Exposed to Python via pybind11 as the bloom module


Key methods: insert, contains, insert_batch, query_batch, benchmark, getMemoryBits
3.2 ML Oracle
* Vectorizer: HashingVectorizer (scikit-learn) — stateless, no vocabulary storage, configurable n_features
* Model: Logistic Regression (max_iter=1000)
* Input: raw URL string
* Output: probability of being a bad URL
3.3 Pipeline (Python)
train/test split (70/30, random_state=42)


    ↓


HashingVectorizer → sparse feature matrix


    ↓


LogisticRegression.fit(X_train, y_train)


    ↓


predict_proba on train → find FN URLs → insert into backup BF


    ↓


predict_proba on test → route negatives through backup BF


    ↓


compute FPR, FNR, memory, latency


________________


4. Experimental Setup
4.1 Grid Search Parameters
Parameter
	Values
	n_features
	1024, 2048, 4096, 16384
	threshold
	0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
	backup_fpr
	0.001, 0.01, 0.05, 0.1, 0.2
	

Total configs: 4 × 8 × 5 = 160 configurations


Model training is cached per n_features (4 model fits total), making the sweep efficient.
4.2 Metrics
System FPR: fraction of good URLs flagged as bad
System FNR: fraction of bad URLs missed by the system
Total memory: model weights (n_features × 64 bits) + BF bits
Latency: model inference time + BF query time, measured in nanoseconds via perf_counter_ns and high_resolution_clock


________________


5. Results
Best configuration per target FPR (minimum memory subject to system_fpr <= target):


Target FPR
	n_features
	Threshold
	Backup FPR
	System FPR
	System FNR
	Learned Memory (MB)
	Standard Memory (MB)
	Memory Reduction
	0.1
	1024
	0.4
	0.05
	0.082
	0.263
	0.0206
	0.0302
	31.8%
	0.05
	1024
	0.7
	0.05
	0.048
	0.416
	0.0279
	0.0393
	29.0%
	0.01
	1024
	0.9
	0.01
	0.0086
	0.596
	0.0510
	0.0605
	15.7%
	0.005
	4096
	0.8
	0.001
	0.0044
	0.384
	0.0721
	0.0696
	-3.6%
	Note: Memory figures for the learned BF include both model weights and BF bits, representing total system cost. The standard BF has no model component — comparisons should be read as total deployment memory, not BF-to-BF. 
5.1 Key Finding: Memory Crossover at FPR < 0.01
At FPR = 0.005, the learned BF uses more memory than the standard BF. This crossover occurs because:


* Tighter FPR targets require a high classification threshold
* High threshold → more training FNs → larger backup BF
* Simultaneously, n_features must increase to 4096 to achieve the required precision
* Model weight memory (n_features × 64 bits) dominates and erases memory savings


This quantifies a clear operating regime boundary: the learned BF only wins on memory when FPR ≥ 0.01 for this dataset.
5.2 Latency
Component
	Avg Latency (ns/query)
	Model alone
	~90–94
	BF alone
	~105–106
	Total system
	~178–194
	Standard BF
	~108–110
	

The learned system incurs a ~1.7x latency overhead vs the standard BF. This is expected — it runs both a model and a BF sequentially, whereas the standard BF is a single lookup. The model's lower memory footprint at relaxed FPR targets may improve CPU cache residency, but this was not directly measured.


________________


6. Limitations and Honest Discussion
6.1 FNR Is A Model Problem, Not A BF Problem
The system FNR (26–60%) reflects the model's generalization error on the test set, not a flaw in the BF. The BF's contract is:


If a bad URL was seen in training AND the model missed it → the BF will catch it


The BF cannot catch bad URLs the model misses at test time — that is by design. BFs are membership data structures; they memorize, they do not generalize. The responsibility for test-time bad URLs lies entirely with the ML oracle.
6.2 Train/Test Overlap Assumption
The random 70/30 split represents a worst-case evaluation for the backup BF — by construction, train and test bad URL sets are disjoint, meaning the backup BF cannot recover any test-time false negatives. This is intentional: it isolates the model's generalization ability as the sole performance driver, ensuring reported FNR reflects true out-of-distribution performance rather than memorization of seen URLs. Simulating overlap (e.g. re-sampling train bad URLs into the test set) would constitute data leakage, conflating memorization with generalization and producing misleading FNR estimates.


In realistic deployment — such as a web proxy where known malicious URLs are repeatedly queried — train/test overlap is naturally high, and the backup BF would meaningfully reduce system FNR beyond what the model achieves alone. Our evaluation therefore reports a conservative lower bound on system performance, with the backup BF's full contribution realised only under production query distributions. 
6.3 Oracle Strength
Logistic Regression on hashed character features is a weak oracle. A stronger model (e.g. TF-IDF + LR, or a character-level neural model) would produce fewer training FNs, shrink the backup BF further, and reduce system FNR. This is the primary avenue for improving results.


________________


7. Future Work
* Temporal split evaluation: Use earlier URLs for training and later URLs for testing to simulate realistic deployment overlap
* Stronger oracle: Replace LR with a character n-gram TF-IDF model or small MLP
* Validation-set threshold selection: Follow Kraska et al. more faithfully — tune threshold on a validation split to provably bound model FNR, then insert validation FNs into the backup BF
* Direct latency measurement: Time query_batch directly rather than using a separate benchmark pass
* SIMD/bitwise optimization: The C++ BF bit array could use SIMD popcount for faster bulk queries


________________


8. What The Project Demonstrates
Despite the limitations above, the project makes the following concrete contributions:


* A working C++17 Bloom Filter with xxHash64 double hashing, exposed to Python via pybind11
* A complete learned BF pipeline integrating a scikit-learn ML oracle with a custom C++ BF
* A systematic benchmark across 160 configurations quantifying the memory-FPR-latency tradeoff
* An empirically identified crossover point (FPR < 0.01) where learned BF loses its memory advantage
* A 1.7x latency characterization of the learned vs standard architecture
* Clear articulation of where the original learned BF assumptions break down under random splitting

9. Project Website
https://krish-goradia.github.io/learned_bloom_filter/