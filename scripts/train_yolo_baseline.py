"""Reproducible Ultralytics baseline trainer for ESWA QCC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--weights", default=None, help="Official pretrained .pt checkpoint, if available")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": args.run_name,
        "detector": args.run_name.split("_")[0],
        "dataset_yaml": str(Path(args.data).resolve()),
        "dataset_yaml_sha256": sha256(Path(args.data)),
        "model_yaml": args.model_yaml,
        "initialization": "official_coco" if args.weights else "random",
        "pretrained": bool(args.weights),
        "source_weights": args.weights,
        "source_weights_sha256": sha256(Path(args.weights)) if args.weights else None,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "workers": args.workers,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model = YOLO(args.weights or args.model_yaml)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=args.workers,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        pretrained=bool(args.weights),
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        # Disable Ultralytics AMP self-check: it downloads an unrelated model from GitHub.
        # GitHub Releases are unreachable from the current cloud instance.
        amp=False,
        cache=False,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
        cos_lr=True,
        patience=0,
    )
    weights = out_dir / "train" / "weights" / "best.pt"
    if not weights.exists():
        raise RuntimeError(f"No best checkpoint produced: {weights}")
    manifest["best_checkpoint"] = str(weights)
    manifest["best_checkpoint_sha256"] = sha256(weights)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
