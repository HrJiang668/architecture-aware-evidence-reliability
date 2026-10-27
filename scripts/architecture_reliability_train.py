"""Architecture-specific reliability training from frozen candidate caches.

This script keeps the shared validity objective while making the evidence path
explicit: DER for dense detectors, RER for ROI detectors, and QCC for query
detectors.  It is intentionally post-hoc: detector checkpoints and candidate
boxes are frozen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from torch import nn


def score_metrics(score: np.ndarray, label: np.ndarray, raw: np.ndarray | None = None) -> dict:
    score = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1 - 1e-6)
    label = np.asarray(label, dtype=np.float64)
    order = np.argsort(-score)
    y = label[order]
    coverage = np.linspace(0.1, 1.0, 19)
    risk = [1.0 - float(y[: max(1, int(np.ceil(c * len(y))))].mean()) for c in coverage]
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (score >= lo) & ((score < hi) if hi < 1 else (score <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(label[mask].mean()) - float(score[mask].mean()))
    out = {
        "auroc": float(roc_auc_score(label, score)),
        "aupr": float(average_precision_score(label, score)),
        "aurc": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(risk, coverage) / (coverage[-1] - coverage[0])),
        "brier": float(brier_score_loss(label, score)),
        "ece15": float(ece),
    }
    for retention in (0.90, 0.95):
        pos = np.sort(score[label == 1])
        threshold = pos[max(0, int(np.floor((1 - retention) * len(pos))))] if len(pos) else 1.0
        accepted = score >= threshold
        fp = int(((label == 0) & accepted).sum())
        out[f"fp_at_tp{int(retention * 100)}"] = fp
        if raw is not None:
            raw_pos = np.sort(np.asarray(raw)[label == 1])
            raw_threshold = raw_pos[max(0, int(np.floor((1 - retention) * len(raw_pos))))] if len(raw_pos) else 1.0
            raw_fp = int(((label == 0) & (np.asarray(raw) >= raw_threshold)).sum())
            out[f"raw_fp_at_tp{int(retention * 100)}"] = raw_fp
            out[f"fp_reduction_at_tp{int(retention * 100)}"] = float((raw_fp - fp) / max(raw_fp, 1))
    return out


def temperature(train_score, train_label, val_score):
    s = np.clip(train_score, 1e-5, 1 - 1e-5)
    logits = torch.tensor(np.log(s / (1 - s)), dtype=torch.float32)
    y = torch.tensor(train_label, dtype=torch.float32)
    t = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits / t.clamp(0.05, 10), y)
        loss.backward()
        return loss

    opt.step(closure)
    v = np.log(np.clip(val_score, 1e-5, 1 - 1e-5) / (1 - np.clip(val_score, 1e-5, 1 - 1e-5)))
    return 1 / (1 + np.exp(-v / float(t.detach().clamp(0.05, 10))))


def platt(train_score, train_label, val_score):
    x = np.log(np.clip(train_score, 1e-5, 1 - 1e-5) / (1 - np.clip(train_score, 1e-5, 1 - 1e-5))).reshape(-1, 1)
    xv = np.log(np.clip(val_score, 1e-5, 1 - 1e-5) / (1 - np.clip(val_score, 1e-5, 1 - 1e-5))).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(x, train_label.astype(int))
    return model.predict_proba(xv)[:, 1]


def isotonic(train_score, train_label, val_score):
    model = IsotonicRegression(y_min=1e-5, y_max=1 - 1e-5, out_of_bounds="clip")
    model.fit(np.asarray(train_score), np.asarray(train_label))
    return model.predict(np.asarray(val_score))


class DenseEvidence(nn.Module):
    def __init__(self, groups, nc, mode="full"):
        super().__init__()
        self.mode = mode
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(int(g)), nn.Linear(int(g), 96), nn.GELU()) for g in groups])
        self.attn = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 1))
        extra = 1 + 1 + 7 + nc + (8 if mode == "full" else 0)
        self.head = nn.Sequential(nn.LayerNorm(96 + extra), nn.Linear(96 + extra, 192), nn.GELU(), nn.Dropout(.1), nn.Linear(192, 1))

    def forward(self, h, score, geom, context, onehot):
        chunks = torch.split(h, [m[1].in_features for m in self.proj], dim=1)
        stages = torch.stack([p(x) for p, x in zip(self.proj, chunks)], dim=1)
        w = torch.softmax(self.attn(stages).squeeze(-1), dim=1)
        evidence = (stages * w.unsqueeze(-1)).sum(1)
        entropy = -(score.clamp(1e-6, 1 - 1e-6) * score.clamp(1e-6, 1 - 1e-6).log() + (1 - score.clamp(1e-6, 1 - 1e-6)) * (1 - score.clamp(1e-6, 1 - 1e-6)).log())
        fields = [evidence, score, entropy, geom, onehot]
        if self.mode == "full":
            fields.append(context)
        return self.head(torch.cat(fields, 1)).squeeze(1)


class RoiEvidence(nn.Module):
    def __init__(self, hidden, nc, mode="full"):
        super().__init__()
        self.mode = mode
        extra = 1 + 1 + 7 + nc + (8 if mode == "full" else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden + extra), nn.Linear(hidden + extra, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))

    def forward(self, h, score, geom, context, onehot):
        p = score.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        fields = [h, score, entropy, geom, onehot]
        if self.mode == "full":
            fields.append(context)
        return self.head(torch.cat(fields, 1)).squeeze(1)


class QueryQCC(nn.Module):
    def __init__(self, groups, nc, mode="full"):
        super().__init__()
        self.mode = mode
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(int(g)), nn.Linear(int(g), 128), nn.GELU()) for g in groups])
        self.traj = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True, norm_first=True), 1)
        extra = 128 + 1 + 1 + 7 + nc + (8 if mode in ("competition", "full") else 0)
        self.head = nn.Sequential(nn.LayerNorm(extra), nn.Linear(extra, 192), nn.GELU(), nn.Dropout(.1), nn.Linear(192, 1))

    def forward(self, h, score, geom, context, onehot):
        chunks = torch.split(h, [m[1].in_features for m in self.proj], dim=1)
        stages = torch.stack([p(x) for p, x in zip(self.proj, chunks)], dim=1)
        final = stages[:, -1]
        if self.mode in ("final", "competition"):
            evidence = final
        else:
            states = self.traj(stages)
            evidence = states.mean(1)
        p = score.clamp(1e-6, 1 - 1e-6)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        fields = [evidence, score, entropy, geom, onehot]
        if self.mode in ("competition", "full"):
            fields.append(context)
        return self.head(torch.cat(fields, 1)).squeeze(1)


def load(path):
    return {k: v for k, v in np.load(path).items()}


def image_level_split(train: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split whole images so candidates from one image never cross the split."""
    groups = train["image"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), train["label"], groups=groups))


def train_model(model, train, val, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    y = torch.tensor(train["label"], dtype=torch.float32, device=device)
    tr, cal = image_level_split(train, seed)
    rng = np.random.default_rng(seed)
    pos = max(1, int(train["label"][tr].sum())); neg = max(1, len(tr) - pos)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nc = int(max(train["cls"].max(), val["cls"].max()) + 1)
    one_train = np.eye(nc, dtype=np.float32)[train["cls"]]
    one_val = np.eye(nc, dtype=np.float32)[val["cls"]]
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    for _ in range(epochs):
        rng.shuffle(tr)
        for start in range(0, len(tr), 2048):
            ix = tr[start:start + 2048]
            opt.zero_grad()
            args = (torch.tensor(train["hidden"][ix], dtype=torch.float32, device=device),
                    torch.tensor(train["score"][ix, None], dtype=torch.float32, device=device),
                    torch.tensor(train["geometry"][ix], dtype=torch.float32, device=device),
                    torch.tensor(train["context"][ix], dtype=torch.float32, device=device),
                    torch.tensor(one_train[ix], dtype=torch.float32, device=device))
            out = model(*args)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y[ix], pos_weight=pos_weight)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        args = (torch.tensor(val["hidden"], dtype=torch.float32, device=device),
                torch.tensor(val["score"][:, None], dtype=torch.float32, device=device),
                torch.tensor(val["geometry"], dtype=torch.float32, device=device),
                torch.tensor(val["context"], dtype=torch.float32, device=device),
                torch.tensor(one_val, dtype=torch.float32, device=device))
        val_pred = torch.sigmoid(model(*args)).cpu().numpy()
        cal_args = (torch.tensor(train["hidden"][cal], dtype=torch.float32, device=device),
                    torch.tensor(train["score"][cal, None], dtype=torch.float32, device=device),
                    torch.tensor(train["geometry"][cal], dtype=torch.float32, device=device),
                    torch.tensor(train["context"][cal], dtype=torch.float32, device=device),
                    torch.tensor(one_train[cal], dtype=torch.float32, device=device))
        cal_pred = torch.sigmoid(model(*cal_args)).cpu().numpy()
    return val_pred, cal_pred, train["score"][cal].astype(np.float32), train["label"][cal].astype(np.float32)


def select_anchor_alpha(raw, head, label):
    def logit(x):
        x = np.clip(x, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))
    def fp90(score):
        pos = np.sort(score[label == 1])
        threshold = pos[max(0, int(np.floor(0.1 * len(pos))))] if len(pos) else 1.0
        return int(((label == 0) & (score >= threshold)).sum())
    raw_fp = fp90(raw)
    best = (0.0, roc_auc_score(label, raw))
    for alpha in np.linspace(0.0, 1.0, 21):
        score = 1 / (1 + np.exp(-((1 - alpha) * logit(raw) + alpha * logit(head))))
        value = roc_auc_score(label, score)
        if fp90(score) <= raw_fp and value > best[1]: best = (float(alpha), float(value))
    return best[0]


def anchor(raw, head, alpha):
    def logit(x):
        x = np.clip(x, 1e-5, 1 - 1e-5)
        return np.log(x / (1 - x))
    return 1 / (1 + np.exp(-((1 - alpha) * logit(raw) + alpha * logit(head))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True); ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--family", choices=["dense", "roi", "query"], required=True)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--primary-only", action="store_true", help="Train only the pre-registered primary native-evidence branch.")
    ap.add_argument("--query-primary", choices=["final", "competition"], default="final")
    args = ap.parse_args(); started = time.perf_counter(); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train, val = load(args.train_cache), load(args.val_cache)
    groups = train.get("hidden_group_sizes")
    if groups is None: groups = np.asarray([train["hidden"].shape[1]], dtype=np.int32)
    nc = int(max(train["cls"].max(), val["cls"].max()) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = val["score"].astype(np.float32)
    _, cal = image_level_split(train, args.seed)
    rng = np.random.default_rng(args.seed)
    result = {"method_spec": {"family": args.family, "groups": groups.tolist(), "seed": args.seed, "detector_frozen": True, "matching": "class-aware one-to-one IoU=0.50"}}
    result["raw"] = score_metrics(raw, val["label"])
    result["temperature"] = score_metrics(temperature(train["score"][cal], train["label"][cal], raw), val["label"], raw)
    result["platt"] = score_metrics(platt(train["score"][cal], train["label"][cal], raw), val["label"], raw)
    result["isotonic"] = score_metrics(isotonic(train["score"][cal], train["label"][cal], raw), val["label"], raw)
    if args.family == "dense":
        variants = [("score_mlp", "metadata"), ("native_feature", "feature"), ("ours_dense", "full")]
        make = lambda mode: DenseEvidence(groups, nc, mode)
    elif args.family == "roi":
        variants = [("score_mlp", "metadata"), ("native_feature", "feature"), ("ours_roi", "full")]
        make = lambda mode: RoiEvidence(int(train["hidden"].shape[1]), nc, mode)
    else:
        variants = [("score_mlp", "metadata"), ("final_query", "final"), ("trajectory_only", "trajectory"), ("competition_only", "competition"), ("ours_qcc", "full")]
        make = lambda mode: QueryQCC(groups, nc, mode)
    if args.primary_only:
        if args.family == "dense": variants = [("native_feature", "feature")]
        elif args.family == "roi": variants = [("native_feature", "feature")]
        else:
            variants = [("competition_only", "competition")] if args.query_primary == "competition" else [("final_query", "final")]
    # score-only baseline is intentionally represented by the raw/temperature rows;
    # the learned score MLP is implemented as a small metadata head below.
    for name, mode in variants:
        if name == "score_mlp":
            xtr = np.column_stack((train["score"], train["geometry"], np.eye(nc)[train["cls"]], train["context"][:, :0]))
            xva = np.column_stack((val["score"], val["geometry"], np.eye(nc)[val["cls"]], val["context"][:, :0]))
            model = nn.Sequential(nn.LayerNorm(xtr.shape[1]), nn.Linear(xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 1)).to(device)
            y = torch.tensor(train["label"], dtype=torch.float32, device=device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            tr, _ = image_level_split(train, args.seed)
            pos = max(1, int(train["label"][tr].sum())); neg = max(1, len(tr) - pos)
            pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
            for _ in range(args.epochs):
                rng.shuffle(tr)
                for start in range(0, len(tr), 4096):
                    ix = tr[start:start + 4096]; opt.zero_grad()
                    logits = model(torch.tensor(xtr[ix], dtype=torch.float32, device=device)).squeeze(1)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y[ix], pos_weight=pos_weight)
                    loss.backward(); opt.step()
            with torch.no_grad(): pred = torch.sigmoid(model(torch.tensor(xva, dtype=torch.float32, device=device))).cpu().numpy().ravel()
        else:
            pred, cal_pred, cal_raw, cal_label = train_model(make(mode), train, val, args.seed, args.epochs, device)
            result[name] = score_metrics(pred, val["label"], raw)
            np.save(out / f"{name}_scores.npy", pred)
            alpha = select_anchor_alpha(cal_raw, cal_pred, cal_label)
            anchored = anchor(val["score"], pred, alpha)
            result[f"anchor_{name}"] = score_metrics(anchored, val["label"], raw)
            result[f"anchor_{name}"]["anchor_alpha"] = alpha
            np.save(out / f"anchor_{name}_scores.npy", anchored)
            continue
        result[name] = score_metrics(pred, val["label"], raw)
        np.save(out / f"{name}_scores.npy", pred)
    result["elapsed_seconds"] = time.perf_counter() - started
    (out / "reliability_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
