import numpy as np


def compute_metrics(results, labels):
    falsePos = 0
    falseNeg = 0
    totalGood = 0
    totalBad = 0

    for r, y in zip(results, labels):
        if y == 1:  # good URL (in-set / member)
            totalGood += 1
            if not r:
                falseNeg += 1  # good URL blocked → FN
        else:       # bad URL (out-of-set / non-member)
            totalBad += 1
            if r:
                falsePos += 1  # bad URL passed → FP

    fpr = falsePos / totalBad if totalBad else 0
    fnr = falseNeg / totalGood if totalGood else 0

    return fpr, fnr


def compute_memory(n_features, bf_bits):
    model_bits = n_features * 64
    return model_bits + bf_bits

