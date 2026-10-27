"""Torchvision Faster R-CNN / RetinaNet training on canonical COCO manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)


class CocoDataset(Dataset):
    def __init__(self, annotation: Path, image_root: Path | None = None):
        self.coco = COCO(str(annotation))
        self.image_root = image_root
        self.ids = sorted(self.coco.getImgIds())
        self.category_ids = sorted(self.coco.getCatIds())
        # Torchvision reserves class 0 for background.  Medical COCO files
        # commonly use category id 0, so preserve an explicit reversible map.
        self.category_to_label = {category_id: index + 1 for index, category_id in enumerate(self.category_ids)}
        self.label_to_category = {label: category_id for category_id, label in self.category_to_label.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        record = self.coco.loadImgs([image_id])[0]
        image_path = Path(record["file_name"])
        if not image_path.is_absolute() and self.image_root is not None:
            image_path = self.image_root / image_path
        image = Image.open(image_path).convert("RGB")
        image = torchvision.transforms.functional.pil_to_tensor(image).float() / 255.0
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=None))
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.category_to_label[ann["category_id"]])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        return image, target


def collate(batch):
    return tuple(x[0] for x in batch), tuple(x[1] for x in batch)


def build_model(name: str, num_classes: int, min_size: int, max_size: int, pretrained: bool = False):
    if name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.transform.min_size = (min_size,)
        model.transform.max_size = max_size
        return model
    if name == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
        )
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        model.transform.min_size = (min_size,)
        model.transform.max_size = max_size
        return model
    raise ValueError(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretrained_weight_path(detector: str) -> Path | None:
    """Return the official cached COCO checkpoint after torchvision constructs it."""
    filenames = {
        "fasterrcnn": "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "retinanet": "retinanet_resnet50_fpn_coco-eeacb38b.pth",
    }
    candidate = Path(torch.hub.get_dir()) / "checkpoints" / filenames[detector]
    return candidate if candidate.is_file() else None


@torch.inference_mode()
def evaluate(model, loader, dataset: CocoDataset) -> dict[str, float]:
    """Evaluate both torchvision baselines with the canonical COCO protocol."""
    model.eval()
    predictions = []
    for images, targets in loader:
        outputs = model([image.cuda(non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            for box, score, label in zip(output["boxes"].detach().cpu().numpy(),
                                         output["scores"].detach().cpu().numpy(),
                                         output["labels"].detach().cpu().numpy()):
                x1, y1, x2, y2 = map(float, box)
                predictions.append({"image_id": int(target["image_id"].item()), "category_id": int(dataset.label_to_category[int(label)]),
                                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
    if not predictions:
        return {"map50_95": 0.0, "map50": 0.0}
    evaluator = COCOeval(dataset.coco, dataset.coco.loadRes(predictions), "bbox")
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["fasterrcnn", "retinanet"], required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--train-image-root", type=Path, default=None)
    ap.add_argument("--val-image-root", type=Path, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--min-size", type=int, default=640)
    ap.add_argument("--max-size", type=int, default=1024)
    ap.add_argument("--pretrained", action="store_true",
                    help="Use torchvision COCO weights when already available or when network download is acceptable.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_json = Path(args.train_json); val_json = Path(args.val_json)
    train_set = CocoDataset(train_json, args.train_image_root)
    val_set = CocoDataset(val_json, args.val_image_root)
    categories = train_set.coco.loadCats(train_set.coco.getCatIds())
    num_classes = len(categories) + 1
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    model = build_model(args.detector, num_classes, args.min_size, args.max_size, args.pretrained).cuda()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs * 2 // 3, 1), gamma=0.1)
    source_path = pretrained_weight_path(args.detector)
    manifest = {"detector": args.detector,
                "initialization": "torchvision_coco" if args.pretrained else "random_initialization",
                "torchvision": torchvision.__version__,
                "num_classes": num_classes, "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
                "min_size": args.min_size, "max_size": args.max_size,
                "category_to_label": train_set.category_to_label,
                "train_json": str(train_json.resolve()), "train_json_sha256": sha256(train_json),
                "val_json": str(val_json.resolve()), "val_json_sha256": sha256(val_json),
                "source_weights": str(source_path) if source_path else None,
                "source_weights_sha256": sha256(source_path) if source_path else None}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (out / "train_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "lr", "seconds", "map50", "map50_95"]); writer.writeheader()
        best = -float("inf")
        best_epoch: int | None = None
        best_metrics: dict[str, float] | None = None
        for epoch in range(1, args.epochs + 1):
            started = time.time(); model.train(); losses = []
            for images, targets in train_loader:
                images = [x.cuda(non_blocking=True) for x in images]
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
                loss = sum(model(images, targets).values())
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step(); mean_loss = float(np.mean(losses)); elapsed = time.time() - started
            metrics = evaluate(model, val_loader, val_set)
            writer.writerow({"epoch": epoch, "loss": mean_loss, "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed, **metrics}); handle.flush()
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "manifest": manifest}, out / "last.pt")
            if metrics["map50_95"] > best:
                best = metrics["map50_95"]
                best_epoch = epoch
                best_metrics = metrics
                torch.save({"epoch": epoch, "model": model.state_dict(), "manifest": manifest, "metrics": metrics}, out / "best.pt")
            print(f"epoch={epoch}/{args.epochs} loss={mean_loss:.6f} map50={metrics['map50']:.5f} map50_95={metrics['map50_95']:.5f} seconds={elapsed:.1f}", flush=True)

    manifest["best_checkpoint"] = str(out / "best.pt")
    manifest["best_checkpoint_sha256"] = sha256(out / "best.pt")
    manifest["best_epoch"] = best_epoch
    manifest["best_metrics"] = best_metrics
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
