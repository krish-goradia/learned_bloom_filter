import numpy as np
import time
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import bloom
from utils import compute_metrics, compute_memory

def prepare_model(train_df, test_df, n_features):

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        
    )

    # vectorizer = HashingVectorizer(
    # analyzer="char",
    # ngram_range=(3,5),
    # n_features=n_features,
    # alternate_sign=False
    # )
    
    X_train = vectorizer.fit_transform(train_df['url'])
    X_test = vectorizer.transform(test_df['url'])

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    probs_train = model.predict_proba(X_train)[:, 1]

    model_start_ns = time.perf_counter_ns()
    probs_test = model.predict_proba(X_test)[:, 1]
    model_total_latency_ns = time.perf_counter_ns() - model_start_ns
    print(X_test.shape)
    print(X_test.nnz)
    print(X_test.nnz / X_test.shape[0])

    test_count = len(test_df)
    if test_count > 0 and model_total_latency_ns > 0:
        model_avg_latency_ns = model_total_latency_ns / test_count
        model_throughput_qps = test_count / (model_total_latency_ns / 1e9)
    else:
        model_avg_latency_ns = 0.0
        model_throughput_qps = 0.0

    return {
        "probs_train": probs_train,
        "probs_test": probs_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_urls": train_df['url'].tolist(),
        "test_urls": test_df['url'].tolist(),
        "model_total_latency_ns": float(model_total_latency_ns),
        "model_avg_latency_ns": model_avg_latency_ns,
        "model_throughput_qps": model_throughput_qps
    }

def run_config(precomp, n_features, threshold, backup_fpr):

    probs_train = precomp["probs_train"]
    probs_test = precomp["probs_test"]
    y_train = precomp["y_train"]
    y_test = precomp["y_test"]
    train_urls = precomp["train_urls"]
    test_urls = precomp["test_urls"]
    model_total_latency_ns = precomp["model_total_latency_ns"]
    model_avg_latency_ns = precomp["model_avg_latency_ns"]
    model_throughput_qps = precomp["model_throughput_qps"]
    preds_train = (probs_train >= threshold).astype(int)
    fn_mask = (y_train == 1) & (preds_train == 0)

    fn_urls = [u for u, m in zip(train_urls, fn_mask) if m]
    learned_bf = bloom.BloomFilter(len(fn_urls), backup_fpr)
    learned_bf.insert_batch(fn_urls)


    model_preds = (probs_test >= threshold)

    bf_inputs = [u for u, mp in zip(test_urls, model_preds) if not mp]
    bf_outputs = learned_bf.query_batch(bf_inputs)

    if bf_inputs:
        bf_stats = learned_bf.benchmark(bf_inputs, len(fn_urls))
        bf_avg_latency_ns = bf_stats.avg_latency_ns
        bf_throughput_qps = bf_stats.throughput_qps
    else:
        bf_avg_latency_ns = 0.0
        bf_throughput_qps = 0.0

    bf_total_latency_ns = bf_avg_latency_ns * len(bf_inputs)
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
    bf_bits = learned_bf.getMemoryBits()
    total_memory = compute_memory(n_features, bf_bits)

    return {
        "n_features": n_features,
        "threshold": threshold,
        "backup_fpr": backup_fpr,
        "fn_count": len(fn_urls),
        "bf_queries": len(bf_inputs),
        "system_fpr": system_fpr,
        "system_fnr": system_fnr,
        "model_total_latency_ns": model_total_latency_ns,
        "model_avg_latency_ns": model_avg_latency_ns,
        "model_throughput_qps": model_throughput_qps,
        "bf_avg_latency_ns": bf_avg_latency_ns,
        "bf_total_latency_ns": bf_total_latency_ns,
        "bf_throughput_qps": bf_throughput_qps,
        "total_system_latency_ns": total_system_latency_ns,
        "total_system_avg_latency_ns": total_system_avg_latency_ns,
        "total_system_throughput_qps": total_system_throughput_qps,
        "bf_bits": bf_bits/ (8 * 1024 * 1024),
        "total_memory_mb": total_memory/ (8 * 1024 * 1024)
    }

def run_standard_bf(train_df, test_df, target_fpr):

    train_bad = train_df.loc[train_df['label'] == 1, 'url'].tolist()

    std_bf = bloom.BloomFilter(len(train_bad), target_fpr)
    std_bf.insert_batch(train_bad)

    test_urls = test_df['url'].tolist()
    test_labels = test_df['label'].tolist()

    results = std_bf.query_batch(test_urls)

    std_stats = std_bf.benchmark(test_urls, len(train_bad))

    from utils import compute_metrics
    fpr, _ = compute_metrics(results, test_labels)

    memory = std_bf.getMemoryBits()/ (8 * 1024 * 1024)

    avg_latency_ns = std_stats.avg_latency_ns
    throughput_qps = std_stats.throughput_qps

    return fpr, memory, avg_latency_ns, throughput_qps