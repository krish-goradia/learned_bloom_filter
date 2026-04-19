import pandas as pd
from sklearn.model_selection import train_test_split
from experiment import prepare_model, run_config, run_standard_bf
from config import *

import os
from tqdm import tqdm


def main():

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
    results_df.to_csv("results/results.csv", index=False)

    valid = results_df[results_df["system_fpr"] <= TARGET_FPR]

    if len(valid) > 0:
        best = valid.sort_values("total_memory_mb").iloc[0]
        print("\nBest Learned BF:")
        print(best)
    else:
        print("\nNo valid config found")

    std_fpr, std_mem = run_standard_bf(train_df, test_df, TARGET_FPR)

    print("\nStandard BF:")
    print("FPR:", std_fpr)
    print("Memory in MB:", std_mem)


if __name__ == "__main__":
    main()