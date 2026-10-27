import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from architecture_reliability_train import load, score_metrics


def one(score, label):
    m = score_metrics(score, label)
    return np.array([m["auroc"], m["aupr"], m["fp_at_tp90"], m["fp_at_tp95"]], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--ours-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    val = load(args.val_cache)
    raw, ours = val["score"].astype(float), np.load(args.ours_scores).astype(float).ravel()
    label, image = val["label"].astype(int), val["image"]
    unique = np.unique(image)
    lookup = {x: np.flatnonzero(image == x) for x in unique}
    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.n):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([lookup[x] for x in chosen])
        r, o = one(raw[ix], label[ix]), one(ours[ix], label[ix])
        diffs.append([o[0] - r[0], o[1] - r[1], (r[2] - o[2]) / max(r[2], 1), (r[3] - o[3]) / max(r[3], 1)])
    d = np.asarray(diffs)
    point = np.array([score_metrics(ours, label)["auroc"] - score_metrics(raw, label)["auroc"],
                      score_metrics(ours, label)["aupr"] - score_metrics(raw, label)["aupr"],
                      (score_metrics(raw, label)["fp_at_tp90"] - score_metrics(ours, label)["fp_at_tp90"]) / max(score_metrics(raw, label)["fp_at_tp90"], 1),
                      (score_metrics(raw, label)["fp_at_tp95"] - score_metrics(ours, label)["fp_at_tp95"]) / max(score_metrics(raw, label)["fp_at_tp95"], 1)])
    result = {"n_images": int(len(unique)), "n_bootstrap": args.n, "seed": args.seed,
              "metrics": ["delta_auroc", "delta_aupr", "fp_reduction_tp90", "fp_reduction_tp95"],
              "point": point.tolist(), "ci95": np.percentile(d, [2.5, 97.5], axis=0).tolist()}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
