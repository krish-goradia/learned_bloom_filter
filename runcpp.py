import pandas as pd
from sklearn.model_selection import train_test_split
from experimentcpp import prepare_model_cpp, run_config_cpp, run_standard_bf
from config import *

import os
from tqdm import tqdm


def main():

    pd.set_option("display.float_format", "{:.6f}".format)

    df = pd.read_csv("dataset/urldata.csv")
    df['label'] = df['label'].map({'good': 1, 'bad': 0})

    good_df = df[df['label'] == 1]
    bad_df  = df[df['label'] == 0]

    bad_train, bad_test = train_test_split(bad_df, test_size=0.2, random_state=RANDOM_STATE)

    train_df = pd.concat([good_df, bad_train]).reset_index(drop=True)
    test_df  = pd.concat([good_df, bad_test]).reset_index(drop=True)

    results = []

    for nf in tqdm(N_FEATURES, desc="n_features"):

        precomp = prepare_model_cpp(train_df, test_df, nf)

        for th in THRESHOLDS:
            for bf_fpr in BACKUP_FPRS:
                res = run_config_cpp(precomp, nf, th, bf_fpr)
                results.append(res)

    results_df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/results_cpp.csv", index=False, float_format="%.6f")

    valid = results_df[results_df["system_fpr"] <= TARGET_FPR]

    if len(valid) > 0:
        best = valid.sort_values("total_memory_mb").iloc[0]
        print("\nBest Learned BF (C++):")
        print(best.to_string(float_format=lambda x: f"{x:.6f}"))
    else:
        print("\nNo valid config found")

    std_fpr, std_mem, std_latency_ns, std_throughput_qps = run_standard_bf(
        train_df,
        test_df,
        TARGET_FPR
    )

    print("\nStandard BF:")
    print(f"FPR:                    {std_fpr:.6f}")
    print(f"Memory (MB):            {std_mem:.6f}")
    print(f"Avg Latency (ns/query): {std_latency_ns:.6f}")
    print(f"Throughput (queries/s): {std_throughput_qps:.6f}")


if __name__ == "__main__":
    main()