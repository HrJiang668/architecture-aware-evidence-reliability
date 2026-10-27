"""Reproducible RF-DETR Small baseline trainer on adapted COCO data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from rfdetr import RFDETRSmall
import rfdetr


def patch_rfdetr_single_gpu_device_check() -> None:
    """Work around RF-DETR 1.9.2 accepting ``devices=[0]`` then treating it as text."""
    from rfdetr.training import trainer

    original = trainer._requests_multiple_devices

    def accepts_lightning_device_list(devices, accelerator=None):
        if isinstance(devices, (list, tuple)):
            return len(devices) > 1
        return original(devices, accelerator)

    trainer._requests_multiple_devices = accepts_lightning_device_list


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--resume", default=None, help="Full RF-DETR/Lightning checkpoint used for exact resume")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "detector": "RF-DETR Small", "initialization": "official_coco", "source_weights": args.weights,
        "source_weights_sha256": sha256(Path(args.weights)), "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "epochs": args.epochs, "batch": args.batch, "grad_accum": args.grad_accum, "workers": args.workers,
        "seed": args.seed, "rfdetr": getattr(rfdetr, "__version__", "unknown"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
    }
    if args.resume:
        resume_path = Path(args.resume)
        manifest["resume_checkpoint"] = str(resume_path.resolve())
        manifest["resume_checkpoint_sha256"] = sha256(resume_path)
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        manifest["resumed_from_epoch"] = int(resume_payload["epoch"])
        del resume_payload
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    patch_rfdetr_single_gpu_device_check()
    model = RFDETRSmall(pretrain_weights=args.weights)
    model.train(dataset_dir=args.dataset_dir, output_dir=str(out), dataset_file="roboflow", device="cuda:0",
                epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum, num_workers=args.workers,
                checkpoint_interval=10, eval_interval=1, use_ema=True, eval_ema_only=True,
                tensorboard=True, wandb=False, seed=args.seed, resolution=512, early_stopping=False,
                notes=manifest, resume=args.resume)
    candidates = sorted(out.glob("checkpoint_best*.pth"))
    if not candidates:
        raise RuntimeError("RF-DETR did not produce a best checkpoint")
    best = next((p for p in candidates if "ema" in p.name), candidates[0])
    manifest["best_checkpoint"] = str(best)
    manifest["best_checkpoint_sha256"] = sha256(best)
    rows = [row for row in csv.DictReader((out / "metrics.csv").open(encoding="utf-8")) if row.get("val/ema_mAP_50_95")]
    if rows:
        best_row = max(rows, key=lambda row: float(row["val/ema_mAP_50_95"]))
        manifest["best_epoch"] = int(best_row["epoch"])
        manifest["best_metrics"] = {
            "map50": float(best_row["val/ema_mAP_50"]),
            "map50_95": float(best_row["val/ema_mAP_50_95"]),
            "precision": float(best_row["val/precision"]),
            "recall": float(best_row["val/recall"]),
        }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
