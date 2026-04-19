import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import bloom
from utils import compute_metrics, compute_memory

def prepare_model(train_df, test_df, n_features):

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False
    )

    X_train = vectorizer.fit_transform(train_df['url'])
    X_test = vectorizer.transform(test_df['url'])

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs_train = model.predict_proba(X_train)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]

    return {
        "probs_train": probs_train,
        "probs_test": probs_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_urls": train_df['url'].tolist(),
        "test_urls": test_df['url'].tolist()
    }

def run_config(precomp, n_features, threshold, backup_fpr):

    probs_train = precomp["probs_train"]
    probs_test = precomp["probs_test"]
    y_train = precomp["y_train"]
    y_test = precomp["y_test"]
    train_urls = precomp["train_urls"]
    test_urls = precomp["test_urls"]
    preds_train = (probs_train >= threshold).astype(int)
    fn_mask = (y_train == 1) & (preds_train == 0)

    fn_urls = [u for u, m in zip(train_urls, fn_mask) if m]
    learned_bf = bloom.BloomFilter(len(fn_urls), backup_fpr)
    learned_bf.insert_batch(fn_urls)


    model_preds = (probs_test >= threshold)

    bf_inputs = [u for u, mp in zip(test_urls, model_preds) if not mp]
    bf_outputs = learned_bf.query_batch(bf_inputs)
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
        "system_fpr": system_fpr,
        "system_fnr": system_fnr,
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

    from utils import compute_metrics
    fpr, _ = compute_metrics(results, test_labels)

    memory = std_bf.getMemoryBits()/ (8 * 1024 * 1024)

    return fpr, memory