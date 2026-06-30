import numpy as np
import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import bloom
from utils import compute_metrics, compute_memory


def prepare_model(train_df, test_df, n_features, n_trials=20, warmup=3):

    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        n_features=n_features,
    )

    X_train = vectorizer.fit_transform(train_df['url'])

    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    probs_train = model.predict_proba(X_train)[:, 1]

    test_urls  = test_df['url'].tolist()
    test_count = len(test_urls)

    # --- warmup (untimed) ---
    for _ in range(warmup):
        X_test     = vectorizer.transform(test_urls)
        probs_test = model.predict_proba(X_test)[:, 1]

    # --- timed trials: vectorize + predict_proba together, from Python ---
    trial_times_ns = []
    for _ in range(n_trials):
        t0         = time.perf_counter_ns()
        X_test     = vectorizer.transform(test_urls)
        probs_test = model.predict_proba(X_test)[:, 1]
        t1         = time.perf_counter_ns()
        trial_times_ns.append(t1 - t0)

    total_latency_ns  = sum(trial_times_ns) / len(trial_times_ns)   # mean total over batch
    avg_latency_ns    = total_latency_ns / test_count
    throughput_qps    = test_count / (total_latency_ns / 1e9)

    return {
        "probs_train":          probs_train,
        "probs_test":           probs_test,
        "y_train":              y_train,
        "y_test":               y_test,
        "train_urls":           train_df['url'].tolist(),
        "test_urls":            test_urls,
        "model_total_latency_ns": float(total_latency_ns),
        "model_avg_latency_ns":   avg_latency_ns,
        "model_throughput_qps":   throughput_qps,
    }


def run_config(precomp, n_features, threshold, backup_fpr, n_trials=20, warmup=3):
    """
    Learned Bloom Filter pipeline (membership framing):

    1. Model passes URL if P(good) >= threshold.
    2. BF backup stores good URLs the model missed (FNs on train set).
    3. A URL is finally passed iff model passes it OR BF says it's a member.

    FP = bad URL passed (dangerous leakage)
    FN = good URL blocked (false alarm)
    """
    probs_train            = precomp["probs_train"]
    probs_test             = precomp["probs_test"]
    y_train                = precomp["y_train"]
    y_test                 = precomp["y_test"]
    train_urls             = precomp["train_urls"]
    test_urls              = precomp["test_urls"]
    model_total_latency_ns = precomp["model_total_latency_ns"]
    model_avg_latency_ns   = precomp["model_avg_latency_ns"]
    model_throughput_qps   = precomp["model_throughput_qps"]

    # --- build backup BF from training FNs ---
    preds_train = (probs_train >= threshold).astype(int)
    fn_mask     = (y_train == 1) & (preds_train == 0)
    fn_urls     = [u for u, m in zip(train_urls, fn_mask) if m]

    learned_bf = bloom.BloomFilter(len(fn_urls), backup_fpr)
    learned_bf.insert_batch(fn_urls)

    # --- test-time inference ---
    model_preds = (probs_test >= threshold)
    bf_inputs   = [u for u, mp in zip(test_urls, model_preds) if not mp]

    if bf_inputs:
        # warmup (untimed)
        for _ in range(warmup):
            learned_bf.query_batch(bf_inputs)

        # timed trials: BF query_batch, from Python
        bf_trial_times_ns = []
        for _ in range(n_trials):
            t0         = time.perf_counter_ns()
            bf_outputs = learned_bf.query_batch(bf_inputs)
            t1         = time.perf_counter_ns()
            bf_trial_times_ns.append(t1 - t0)

        bf_total_latency_ns = sum(bf_trial_times_ns) / len(bf_trial_times_ns)
        bf_avg_latency_ns   = bf_total_latency_ns / len(bf_inputs)
        bf_throughput_qps   = len(bf_inputs) / (bf_total_latency_ns / 1e9)
    else:
        bf_outputs        = []
        bf_total_latency_ns = 0.0
        bf_avg_latency_ns   = 0.0
        bf_throughput_qps   = 0.0

    # weighted: model runs on all URLs, BF only on blocked fraction
    total_system_latency_ns = model_total_latency_ns + bf_total_latency_ns
    if len(test_urls) > 0 and total_system_latency_ns > 0:
        total_system_avg_latency_ns = total_system_latency_ns / len(test_urls)
        total_system_throughput_qps = len(test_urls) / (total_system_latency_ns / 1e9)
    else:
        total_system_avg_latency_ns = 0.0
        total_system_throughput_qps = 0.0

    # --- combine: pass if model passes OR BF confirms membership ---
    result = model_preds.copy()
    bf_idx = 0
    for i in range(len(result)):
        if not result[i]:
            result[i] = bool(bf_outputs[bf_idx])
            bf_idx += 1

    system_fpr, system_fnr = compute_metrics(result, y_test)

    bf_bits      = learned_bf.getMemoryBits()
    total_memory = compute_memory(n_features, bf_bits)

    return {
        "n_features":                   n_features,
        "threshold":                    threshold,
        "backup_fpr":                   backup_fpr,
        "fn_count":                     len(fn_urls),
        "bf_queries":                   len(bf_inputs),
        "system_fpr":                   system_fpr,
        "system_fnr":                   system_fnr,
        "model_total_latency_ns":       model_total_latency_ns,
        "model_avg_latency_ns":         model_avg_latency_ns,
        "model_throughput_qps":         model_throughput_qps,
        "bf_avg_latency_ns":            bf_avg_latency_ns,
        "bf_total_latency_ns":          float(bf_total_latency_ns),
        "bf_throughput_qps":            bf_throughput_qps,
        "total_system_latency_ns":      total_system_latency_ns,
        "total_system_avg_latency_ns":  total_system_avg_latency_ns,
        "total_system_throughput_qps":  total_system_throughput_qps,
        "bf_bits_mb":                   bf_bits / (8 * 1024 * 1024),
        "total_memory_mb":              total_memory / (8 * 1024 * 1024),
    }


def run_standard_bf(train_df, test_df, target_fpr, n_trials=20, warmup=3):
    """
    Standard Bloom Filter baseline (membership framing):
    Insert all good (in-set) training URLs. Query all test URLs.

    FP = bad URL returned as member (dangerous)
    FN = impossible by BF design (good URLs always found if inserted)
    """
    train_good = train_df.loc[train_df['label'] == 1, 'url'].tolist()

    std_bf = bloom.BloomFilter(len(train_good), target_fpr)
    std_bf.insert_batch(train_good)

    test_urls   = test_df['url'].tolist()
    test_labels = test_df['label'].tolist()

    # warmup (untimed)
    for _ in range(warmup):
        std_bf.query_batch(test_urls)

    # timed trials: BF query_batch, from Python
    trial_times_ns = []
    for _ in range(n_trials):
        t0      = time.perf_counter_ns()
        results = std_bf.query_batch(test_urls)
        t1      = time.perf_counter_ns()
        trial_times_ns.append(t1 - t0)

    total_ns       = sum(trial_times_ns) / len(trial_times_ns)
    avg_latency_ns = total_ns / len(test_urls)
    throughput_qps = len(test_urls) / (total_ns / 1e9)

    fpr, _    = compute_metrics(results, test_labels)
    memory_mb = std_bf.getMemoryBits() / (8 * 1024 * 1024)

    return fpr, memory_mb, avg_latency_ns, throughput_qps
