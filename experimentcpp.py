import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
import bloom
from utils import compute_metrics, compute_memory


# ---------------------------------------------------------------------------
# Core idea: pass the WHOLE batch in one call, let C++ loop internally,
# time the single call from Python, then divide by batch size to get the
# mean per-query latency. This way we pay the Python<->C++ crossing cost
# ONCE for the entire batch, not once per item, and the per-item average
# we report is honest (crossing overhead amortized + real per-item work).
# ---------------------------------------------------------------------------
def time_batched_call(fn, n_items, n_trials=20, warmup=3):
    """
    fn: zero-arg callable that runs ONE full batch call
    n_items: number of queries in that batch
    n_trials: number of repeated batch calls, averaged (mean)
    warmup: untimed calls first, to avoid cold-cache/first-call skew

    Returns: (mean_latency_ns_per_item, throughput_qps, result_of_last_call)
    """
    if n_items == 0:
        return 0.0, 0.0, None

    result = None
    for _ in range(warmup):
        result = fn()

    total_times_s = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        total_times_s.append(t1 - t0)

    mean_total_s = sum(total_times_s) / len(total_times_s)
    avg_latency_ns = (mean_total_s / n_items) * 1e9
    throughput_qps = n_items / mean_total_s if mean_total_s > 0 else 0.0

    return avg_latency_ns, throughput_qps, result


def prepare_model_cpp(train_df, test_df, n_features, n_trials=20):
    """
    C++ fused vectorizer + scorer.
    vec.transform_and_score() is called ONCE with the full batch of test
    URLs; C++ loops internally. We time that single call from Python and
    divide by batch size for the mean per-query latency.
    """
    train_urls = train_df['url'].tolist()
    test_urls  = test_df['url'].tolist()
    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    vec = bloom.Vectorizer(n_features, 3, 3)

    csr = vec.transform(train_urls)
    X_train = csr_matrix(
        (csr.data, csr.indices, csr.indptr),
        shape=(csr.n_rows, csr.n_cols)
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    probs_train = model.predict_proba(X_train)[:, 1]

    weights   = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])

    test_count = len(test_urls)

    def call_fn():
        return vec.transform_and_score(test_urls, weights, intercept)

    model_avg_latency_ns, model_throughput_qps, scored = time_batched_call(
        call_fn, n_items=test_count, n_trials=n_trials
    )
    model_total_latency_ns = model_avg_latency_ns * test_count

    probs_test = np.array(scored.probs)
    auc = roc_auc_score(y_test, probs_test)
    print(f"AUC: {auc:.4f}")
    print(f"[Python-timed] fused vec+score: {model_avg_latency_ns:.1f} ns/query "
          f"(mean over {n_trials} trials, batch={test_count}), {model_throughput_qps:.0f} qps")

    return {
        "probs_train":             probs_train,
        "probs_test":              probs_test,
        "y_train":                 y_train,
        "y_test":                  y_test,
        "train_urls":              train_urls,
        "test_urls":               test_urls,
        "model_total_latency_ns":  float(model_total_latency_ns),
        "model_avg_latency_ns":    model_avg_latency_ns,
        "model_throughput_qps":    model_throughput_qps,
    }


def prepare_model_sklearn(train_df, test_df, n_features, n_trials=20):
    """
    Baseline: sklearn HashingVectorizer + predict_proba, timed the same way
    — one full batch call (vectorize + predict on all test URLs), timed
    from Python, divided by batch size.
    """
    train_urls = train_df['url'].tolist()
    test_urls  = test_df['url'].tolist()
    y_train = train_df['label'].values
    y_test  = test_df['label'].values

    hv = HashingVectorizer(n_features=n_features, analyzer='char', ngram_range=(3, 3))
    X_train = hv.transform(train_urls)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    probs_train = model.predict_proba(X_train)[:, 1]

    test_count = len(test_urls)

    def call_fn():
        X_test = hv.transform(test_urls)
        return model.predict_proba(X_test)[:, 1]

    model_avg_latency_ns, model_throughput_qps, probs_test = time_batched_call(
        call_fn, n_items=test_count, n_trials=n_trials
    )
    model_total_latency_ns = model_avg_latency_ns * test_count

    auc = roc_auc_score(y_test, probs_test)
    print(f"AUC: {auc:.4f}")
    print(f"[Python-timed] sklearn vec+predict: {model_avg_latency_ns:.1f} ns/query "
          f"(mean over {n_trials} trials, batch={test_count}), {model_throughput_qps:.0f} qps")

    return {
        "probs_train":             probs_train,
        "probs_test":              probs_test,
        "y_train":                 y_train,
        "y_test":                  y_test,
        "train_urls":              train_urls,
        "test_urls":               test_urls,
        "model_total_latency_ns":  float(model_total_latency_ns),
        "model_avg_latency_ns":    model_avg_latency_ns,
        "model_throughput_qps":    model_throughput_qps,
    }


def run_config_cpp(precomp, n_features, threshold, backup_fpr, n_trials=20):
    """
    Learned Bloom Filter pipeline. BF query_batch() is called ONCE with all
    blocked URLs as a batch; C++ loops internally. Timed from Python the
    same way as the model above.
    """
    probs_train             = precomp["probs_train"]
    probs_test              = precomp["probs_test"]
    y_train                 = precomp["y_train"]
    y_test                  = precomp["y_test"]
    train_urls              = precomp["train_urls"]
    test_urls               = precomp["test_urls"]
    model_total_latency_ns  = precomp["model_total_latency_ns"]
    model_avg_latency_ns    = precomp["model_avg_latency_ns"]
    model_throughput_qps    = precomp["model_throughput_qps"]

    preds_train = (probs_train >= threshold).astype(int)
    fn_mask     = (y_train == 1) & (preds_train == 0)
    fn_urls     = [u for u, m in zip(train_urls, fn_mask) if m]

    learned_bf = bloom.BloomFilter(len(fn_urls), backup_fpr)
    learned_bf.insert_batch(fn_urls)

    model_preds = (probs_test >= threshold)
    bf_inputs = [u for u, mp in zip(test_urls, model_preds) if not mp]

    def bf_call_fn():
        return learned_bf.query_batch(bf_inputs)

    bf_avg_latency_ns, bf_throughput_qps, bf_outputs = time_batched_call(
        bf_call_fn, n_items=len(bf_inputs), n_trials=n_trials
    )
    bf_total_latency_ns = bf_avg_latency_ns * len(bf_inputs)

    if bf_inputs:
        print(f"[Python-timed] BF query_batch: {bf_avg_latency_ns:.1f} ns/query "
              f"(mean over {n_trials} trials, batch={len(bf_inputs)}), {bf_throughput_qps:.0f} qps")

    total_system_latency_ns = model_total_latency_ns + bf_total_latency_ns
    if len(test_urls) > 0 and total_system_latency_ns > 0:
        total_system_avg_latency_ns = total_system_latency_ns / len(test_urls)
        total_system_throughput_qps = len(test_urls) / (total_system_latency_ns / 1e9)
    else:
        total_system_avg_latency_ns = 0.0
        total_system_throughput_qps = 0.0

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
        "bf_total_latency_ns":          bf_total_latency_ns,
        "bf_throughput_qps":            bf_throughput_qps,
        "total_system_latency_ns":      total_system_latency_ns,
        "total_system_avg_latency_ns":  total_system_avg_latency_ns,
        "total_system_throughput_qps":  total_system_throughput_qps,
        "bf_bits_mb":                   bf_bits / (8 * 1024 * 1024),
        "total_memory_mb":              total_memory / (8 * 1024 * 1024),
    }


def run_standard_bf(train_df, test_df, target_fpr, n_trials=20):
    """
    Standard Bloom Filter baseline. One batch call, timed from Python,
    divided by batch size.
    """
    train_good = train_df.loc[train_df['label'] == 1, 'url'].tolist()

    std_bf = bloom.BloomFilter(len(train_good), target_fpr)
    std_bf.insert_batch(train_good)

    test_urls   = test_df['url'].tolist()
    test_labels = test_df['label'].tolist()

    def call_fn():
        return std_bf.query_batch(test_urls)

    avg_latency_ns, throughput_qps, results = time_batched_call(
        call_fn, n_items=len(test_urls), n_trials=n_trials
    )

    fpr, _    = compute_metrics(results, test_labels)
    memory_mb = std_bf.getMemoryBits() / (8 * 1024 * 1024)

    print(f"[Python-timed] standard BF query_batch: {avg_latency_ns:.1f} ns/query "
          f"(mean over {n_trials} trials, batch={len(test_urls)}), {throughput_qps:.0f} qps")

    return fpr, memory_mb, avg_latency_ns, throughput_qps