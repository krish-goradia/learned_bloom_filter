import pandas as pd
from sklearn.model_selection import train_test_split
from experiment import prepare_model, run_config, run_standard_bf
from config import *

import os
from tqdm import tqdm


def main():

    pd.set_option("display.float_format", "{:.6f}".format)

    df = pd.read_csv("dataset/urldata.csv")
    df['label'] = df['label'].map({'good': 0, 'bad': 1})

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_STATE
    )

    results = []

    for nf in tqdm(N_FEATURES, desc="n_features"):

        precomp = prepare_model(train_df, test_df, nf)

        for th in THRESHOLDS:

            probs_test = precomp["probs_test"]
            model_preds = (probs_test >= th)

            for bf_fpr in BACKUP_FPRS:

                res = run_config(precomp, nf, th, bf_fpr)
                results.append(res)

    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/results.csv", index=False, float_format="%.6f")

    valid = results_df[results_df["system_fpr"] <= TARGET_FPR]

    if len(valid) > 0:
        best = valid.sort_values("total_memory_mb").iloc[0]
        print("\nBest Learned BF:")
        print(best.to_string(float_format=lambda x: f"{x:.6f}"))
    else:
        print("\nNo valid config found")

    std_fpr, std_mem, std_latency_ns, std_throughput_qps = run_standard_bf(
        train_df,
        test_df,
        TARGET_FPR
    )

    print("\nStandard BF:")
    print(f"FPR: {std_fpr:.6f}")
    print(f"Memory in MB: {std_mem:.6f}")
    print(f"Avg Latency (ns/query): {std_latency_ns:.6f}")
    print(f"Throughput (queries/s): {std_throughput_qps:.6f}")


if __name__ == "__main__":
    main()