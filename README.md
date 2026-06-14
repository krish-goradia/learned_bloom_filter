# Learned Bloom Filters — Project Documentation

**Research Supervisor:** Prof. Anirban Dasgupta, IIT Gandhinagar  
**Duration:** January 2026 – April 2026  
**Dataset:** 420K URLs (binary classification: good/bad)

---

## 1. Motivation

A classical Bloom Filter (BF) is a space-efficient probabilistic data structure that answers set-membership queries with a tunable false positive rate (FPR) and a strict **zero false negative** guarantee. It treats every element identically — it has no notion that some elements are structurally more predictable than others.

The **Learned Bloom Filter** (Kraska et al., 2018) replaces part of the BF with a learned model acting as a pre-filter. The model classifies most members correctly; a much smaller backup BF handles the model's training-time mistakes. For structured data like URLs — which have domain patterns — the hypothesis is that the model generalises well enough that the backup BF needs to store far fewer elements than a full BF, yielding memory savings at the same FPR target.

---

## 2. Problem Framing: Membership Testing

The experiment is modelled as a **database membership query**, not a generic binary classifier. The set of **good URLs is the membership set** — the collection we want to answer queries about. Bad URLs are non-members.

| Label | Meaning |
|---|---|
| `good` (label = 1) | In-set member — should be returned as a match |
| `bad`  (label = 0) | Non-member — should not be returned |

**Train/test split:**
- All good URLs appear in **both** train and test sets
- Bad URLs are split 80/20 (train/test), `random_state=42`

This mirrors a realistic database deployment: every known member is in the index. The system is evaluated on its ability to correctly pass members and reject non-members.

**Consequence for FNR:** Because all good URLs are in the training set, the backup BF sees every member the model missed at training time. At test time, those same members are queried again — and the backup BF catches them all. **System FNR is therefore 0.000 across all configurations by construction**, not as an empirical surprise.

**FP and FN in this framing:**
- **FP** = bad URL (non-member) passed through — dangerous leakage, bounded by target FPR
- **FN** = good URL (member) blocked — eliminated by the backup BF since all members were seen at training

---

## 3. System Architecture

### 3.1 Standard Bloom Filter (Baseline)

```
Query URL → Bloom Filter (stores all good/in-set training URLs) → pass / block
```

- **True positive** (member found): correctly passed
- **False positive** (non-member found): incorrectly passed — bounded by target FPR
- **True negative** (non-member not found): correctly blocked
- **False negative**: impossible by BF construction

### 3.2 Learned Bloom Filter

```
Query URL → ML Model → P(member) ≥ threshold?
                YES  → pass
                NO   → query Backup BF → found?
                              YES → pass
                              NO  → block
```

The backup BF is populated with members the model incorrectly blocked on the **training set**. Since all good URLs appear in training, this guarantees FNR = 0 at test time — if the model blocks a member, the BF will catch it.

---

## 4. Implementation

### 4.1 Bloom Filter (C++17)

Built from scratch using:
- **xxHash64** with double hashing: `idx = (h1 + i * h2) % m` for `i in 0..k`
- Bit array stored as `vector<uint64_t>` for cache-friendly access
- Two constructors: direct `(m, k)` or optimal `(n, target_fpr)` using standard formulae:
  - `m = -(n × ln(fpr)) / ln(2)²`
  - `k = (m/n) × ln(2)`
- Exposed to Python via **pybind11** as the `bloom` module

### 4.2 ML Oracle

- **Classifier:** Logistic Regression (`max_iter=1000`, `class_weight='balanced'`)
- **Input:** raw URL string
- **Output:** P(member | URL) — probability the URL is an in-set member
- Threshold applied to decide pass/block; URLs below threshold are routed to the backup BF

### 4.3 Pipeline

```
All good URLs + 80% bad URLs → train set
All good URLs + 20% bad URLs → test set
        ↓
Vectorizer → feature matrix
        ↓
LogisticRegression.fit(X_train, y_train)
        ↓
predict on train → find FN URLs (members model blocked) → insert into backup BF
        ↓
predict on test  → route model negatives through backup BF
        ↓
compute system FPR, FNR, memory, latency
```
### 4.4 Why Logistic Regression?
Logistic Regression was selected because inference reduces to a sparse dot product over active features, giving O(NNZ) evaluation cost. Its parameter count scales linearly with feature dimension, coefficients can be exported directly into a C++ array, and the model requires no runtime dependencies during deployment.

---

## 5. Vectorisation Strategies

Three vectorisation strategies were evaluated, each affecting model accuracy and end-to-end latency differently.

### 5.1 Python HashingVectorizer — Default (Word-Level)

scikit-learn's `HashingVectorizer` with default settings: word-level tokenisation, splitting URLs on whitespace and punctuation. Each URL produces ~3–8 word tokens, resulting in an average of 6 non-zero entries (NNZ) per row in the CSR matrix. Sparse dot-product inference is therefore fast, but word tokens discard all subword structure within path components, subdomains, and query strings.

| Metric | Value |
|---|---|
| Median vec latency | ~7,548 ns/query |
| Median model latency | ~96 ns/query |
| Median total system latency | ~7,693 ns/query |
| Median throughput | ~130K QPS |

### 5.2 Python HashingVectorizer — Character Trigrams

`HashingVectorizer(analyzer='char', ngram_range=(3,3))`. A URL of length L produces L−2 trigrams, capturing subword patterns like `.php`, `/admin/`, or `://x` that word tokens miss entirely. This substantially improves model quality — fewer training FNs, smaller backup BF — but at a steep latency cost.

The root cause is the CSR matrix structure. Character trigrams activate an average of ~160 per row vs  ~6 for word tokens. scikit-learn's `transform` and `predict_proba` operate on the full test batch. The reported ns/query is therefore an **amortised batch average**, not true per-query online latency — a real serving system processing one URL at a time would be significantly slower.

| Metric | Value |
|---|---|
| Median vec latency | ~26,406 ns/query *(amortised batch)* |
| Median model latency | ~202 ns/query |
| Median total system latency | ~26,671 ns/query |
| Median throughput | ~37K QPS |

### 5.3 C++ Vectorizer — Character Trigrams (Fused, Rolling Hash)

The Python implementation was useful for rapid experimentation but measured amortised batch latency through scikit-learn's sparse matrix pipeline. To evaluate true online serving performance, vectorisation and inference were reimplemented in C++, removing Python function-call overhead, avoiding CSR matrix construction for every query, combining vectorisation and scoring into a single pass, and using a rolling hash instead of generating trigram strings. This enabled direct measurement of true per-query latency and significantly improved throughput.

**Rolling hash:** Character trigrams are generated using a rolling hash, allowing O(1) updates per shift without allocating intermediate substring objects.

Key design decisions:

- **Bitmask indexing:** `idx = hash & (n_features - 1)` — requires `n_features` to be a power of two, eliminates modulo entirely
- **Dirty-index pattern:** a flat `float[n_features]` array stays zeroed across queries; only touched indices are tracked and reset, so effective work per URL is O(NNZ)
- **Fused vec+score:** model weights are exported once from Python's `LogisticRegression` as a `double[]` array; the C++ kernel accumulates the logit inline during feature extraction, with no intermediate matrix

| Metric | Value |
|---|---|
| Fused vec+score latency | ~445–461 ns/query *(true per-query)* |
| Median total system latency | ~488 ns/query |
| Median throughput | ~2.0M QPS |
| Speedup vs Python trigram | **~54×** |
| Speedup vs Python default | **~15×** |

---

## 6. Experimental Setup

### 6.1 Grid Search Parameters

| Parameter | Values |
|---|---|
| `n_features` | 1024, 2048, 4096, 16384 |
| `threshold` | 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 |
| `backup_fpr` | 0.001, 0.01, 0.05, 0.1, 0.2 |

Total: 4 × 7 × 5 = **140 configurations** per variant. Model training is cached per `n_features` (4 fits total).

### 6.2 Metrics

- **System FPR:** fraction of bad (non-member) URLs incorrectly passed — the primary constraint
- **System FNR:** fraction of good (member) URLs incorrectly blocked — always 0 by construction (see Section 2)
- **Total memory:** model weights (`n_features × 64` bits) + backup BF bits
- **Latency:** measured via `perf_counter_ns` (Python) and `high_resolution_clock` (C++)

---

## 7. Results

### 7.1 Standard Bloom Filter Baseline

| Target FPR | Actual FPR | Memory (MB) | Avg Latency (ns) | Throughput (QPS) |
|---|---|---|---|---|
| 0.005 | 0.004627 | 0.4533 | 149.2 | 6,704,215 |
| 0.01  | 0.009584 | 0.3940 | 147.2 | 6,795,625 |
| 0.05  | 0.049111 | 0.2563 | 118.9 | 8,410,656 |
| 0.1   | 0.105559 | 0.1970 | 107.7 | 9,285,462 |

### 7.2 Python Default (Word-Level) — Best Config per Target FPR

| Target FPR | n_features | Threshold | Backup FPR | System FPR | Memory (MB) | Std BF (MB) | Reduction |
|---|---|---|---|---|---|---|---|
| 0.005 | 16384 | 0.95 | 0.001 | 0.00436 | 0.3794 | 0.4533 | −16.3% |
| 0.01  | 4096  | 0.95 | 0.001 | 0.00780 | 0.3444 | 0.3940 | −12.6% |
| 0.05  | 4096  | 0.80 | 0.01  | 0.04600 | 0.1372 | 0.2563 | −46.5% |
| 0.1   | 4096  | 0.60 | 0.01  | 0.09994 | 0.0833 | 0.1970 | −57.7% |

System FNR = 0.000 across all configurations.

### 7.3 Python Char Trigram — Best Config per Target FPR

| Target FPR | n_features | Threshold | Backup FPR | System FPR | Memory (MB) | Std BF (MB) | Reduction |
|---|---|---|---|---|---|---|---|
| 0.005 | 16384 | 0.95 | 0.001 | 0.00430 | 0.3210 | 0.4533 | −29.2% |
| 0.01  | 16384 | 0.90 | 0.001 | 0.00846 | 0.2529 | 0.3940 | −35.8% |
| 0.05  | 4096  | 0.80 | 0.01  | 0.03893 | 0.0958 | 0.2563 | −62.6% |
| 0.1   | 4096  | 0.50 | 0.01  | 0.08778 | 0.0538 | 0.1970 | −72.7% |

System FNR = 0.000 across all configurations.

### 7.4 C++ Char Trigram (Fused) — Best Config per Target FPR

| Target FPR | n_features | Threshold | Backup FPR | System FPR | Memory (MB) | Std BF (MB) | Reduction |
|---|---|---|---|---|---|---|---|
| 0.005 | 4096 | 0.99 | 0.001 | 0.00430 | 0.3140 | 0.4533 | −30.7% |
| 0.01  | 4096 | 0.95 | 0.001 | 0.00972 | 0.2078 | 0.3940 | −47.3% |
| 0.05  | 4096 | 0.80 | 0.01  | 0.04349 | 0.0897 | 0.2563 | −65.0% |
| 0.1   | 4096 | 0.50 | 0.01  | 0.09042 | 0.0544 | 0.1970 | −72.4% |

System FNR = 0.000 across all configurations.

### 7.5 Key Findings

**Memory savings grow at relaxed FPR targets.** Tighter FPR targets require higher thresholds, which pushes more members into the backup BF and erodes the memory advantage. At FPR ≤ 0.005 the C++ learned BF saves ~31% vs the standard BF; at FPR ≤ 0.1 savings reach ~72%.

**Feature-space tradeoff.** Increasing n_features beyond 4096 reduced hash collisions and slightly improved oracle quality, but the growth in model memory outweighed the reduction in backup Bloom filter size. As a result, 4096 features was the most memory-efficient operating point across most FPR targets.

**Character trigrams produce fewer training FNs than word tokens.** Although they increase average feature density from ~6 NNZ to ~160 NNZ per URL, they reduce training false negatives from ~138K to ~92K (~33%), directly shrinking the backup Bloom filter and improving memory efficiency.

**C++ delivers true per-query latency; Python numbers are amortised.** The C++ fused kernel is ~54× faster than Python char trigrams and ~15× faster than Python word tokenisation. The standard BF remains faster still (~108–149 ns) since it is a single lookup with no model overhead.

### 7.6 Latency Summary

| Variant | Vec (ns) | Model (ns) | Total System (ns) | QPS |
|---|---|---|---|---|
| Python word (default) | ~7,548 | ~96 | ~7,693 | ~130K |
| Python char trigram | ~26,406 *(amortised)* | ~202 | ~26,671 | ~37K |
| C++ fused trigram | *(fused)* | ~445–461 | ~468–527 | ~2.0M |
| Standard BF (baseline) | — | — | ~108–149 | ~6.7–9.3M |

---

## 8. Limitations and Honest Discussion

### 8.1 FNR = 0 Is a Property of the Split, Not the Model

System FNR is zero because all good URLs appear in the training set. The backup BF memorises every member the model missed at training time, and since those same members appear at test time, they are always recovered. This is not a claim about model quality — a weaker model simply pushes more members into the backup BF, inflating its size.

### 8.2 Python Latency Is Amortised

scikit-learn's `transform` and `predict_proba` batch the entire test set and exploit NumPy/BLAS parallelism. The reported ns/query for Python variants is a lower bound on true single-URL online latency. C++ numbers are genuine per-query measurements and are the only directly comparable figures for a real serving scenario.

---
### 8.3 Memory Accounting

Reported memory includes only the Logistic Regression model weights and the backup Bloom filter. Python runtime overhead, temporary data structures, and scikit-learn internal objects were not included, so the comparison focuses on the memory used by the learned Bloom filter itself rather than language-specific overhead.

## 9. What This Project Demonstrates

- A working C++17 Bloom Filter with xxHash64 double hashing, exposed to Python via pybind11
- Three vectorisation strategies benchmarked end-to-end: Python word tokens, Python char trigrams, and a fused C++ rolling-hash trigram kernel
- Quantification of the amortised vs true latency gap: Python batch inference reports ~26µs/query; the C++ fused kernel measures ~461 ns/query — a ~54× difference in true per-query cost
- A systematic benchmark across 140 configurations per variant characterising the memory–FPR–latency trade-off
- Empirical demonstration that char trigrams produce ~33% fewer training FNs than word tokens, directly translating to better memory savings at the same FPR target
- Clear explanation of why FNR = 0 under this experimental design and what that does and does not imply about model quality
