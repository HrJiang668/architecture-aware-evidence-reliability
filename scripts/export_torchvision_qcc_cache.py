"""Export frozen torchvision detector candidates with FPN visual evidence.

The detector is used only for inference. Candidate labels are class-aware,
one-to-one IoU=.50 matches against the supplied COCO split; no final-split
label is used during reliability-head fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import roi_align


def build_model(detector: str, classes: int, min_size: int, max_size: int) -> torch.nn.Module:
    if detector == "fasterrcnn":
        # A trained experiment checkpoint is loaded below. Explicitly disable
        # torchvision's default backbone download during cache extraction.
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes + 1)
        model.transform.min_size = (min_size,)
        model.transform.max_size = max_size
        return model
    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None)
    head = model.head.classification_head
    channels = head.conv[0][0].in_channels
    # The detector baseline was trained with the shared torchvision class
    # convention (background slot included), unlike native RetinaNet's usual
    # foreground-only convention. Reproduce that exact trained head here.
    model.head.classification_head = RetinaNetClassificationHead(channels, head.num_anchors, classes + 1)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    other = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area + other - inter, 1e-8)


def match(boxes: np.ndarray, cls: np.ndarray, score: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(boxes), dtype=np.float32); best = np.zeros(len(boxes), dtype=np.float32); used: set[int] = set()
    for idx in np.argsort(-score):
        same = np.where(targets[:, 0].astype(int) == int(cls[idx]))[0] if len(targets) else np.empty(0, dtype=int)
        if len(same):
            values = iou(boxes[idx], targets[same, 1:]); winner = int(same[values.argmax()]); best[idx] = values.max()
            if best[idx] >= 0.5 and winner not in used:
                labels[idx] = 1.0; used.add(winner)
    return labels, best


def context(boxes: np.ndarray, scores: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 8), dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2; distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1); norm = max(size)
    width_height = np.maximum(boxes[:, 2:] - boxes[:, :2], 0); areas = width_height[:, 0] * width_height[:, 1]
    xx = np.maximum(0, np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0]))
    yy = np.maximum(0, np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]))
    overlap = xx * yy / np.maximum(areas[:, None] + areas[None, :] - xx * yy, 1e-6); np.fill_diagonal(overlap, 0)
    rival = scores[None, :].repeat(len(scores), axis=0); np.fill_diagonal(rival, 0)
    min_distance = nearest[:, 0] / norm if len(boxes) > 1 else np.ones(len(boxes))
    mean3 = nearest[:, :min(3, len(boxes) - 1)].mean(1) / norm if len(boxes) > 1 else np.ones(len(boxes))
    mean8 = nearest[:, :min(8, len(boxes) - 1)].mean(1) / norm if len(boxes) > 1 else np.ones(len(boxes))
    return np.column_stack((min_distance, mean3, mean8, overlap.max(1), overlap.mean(1), rival.max(1), scores - rival.max(1), (distances < norm / 4).sum(1) / 8)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["fasterrcnn", "retinanet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    coco = COCO(args.coco_json); category_ids = sorted(coco.getCatIds()); category_to_zero = {item: pos for pos, item in enumerate(category_ids)}
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.detector, len(category_ids), args.min_size, args.max_size)
    model.load_state_dict(payload["model"], strict=True); model.cuda().eval()
    state: dict[str, object] = {}
    hooks = [
        model.transform.register_forward_hook(lambda _m, _i, output: state.__setitem__("image_sizes", output[0].image_sizes)),
        model.backbone.register_forward_hook(lambda _m, _i, output: state.__setitem__("features", output)),
    ]
    records = []
    try:
        for number, image_id in enumerate(sorted(coco.getImgIds()), 1):
            image_info = coco.loadImgs([image_id])[0]
            path = Path(image_info["file_name"])
            if not path.is_absolute() and args.image_root is not None:
                path = args.image_root / path
            path = path.resolve()
            tensor = torchvision.transforms.functional.pil_to_tensor(Image.open(path).convert("RGB")).float().div(255).cuda()
            with torch.inference_mode():
                output = model([tensor])[0]
            mask = output["scores"] >= args.conf
            boxes = output["boxes"][mask].detach().cpu().numpy().astype(np.float32)
            scores = output["scores"][mask].detach().cpu().numpy().astype(np.float32)
            labels_one = output["labels"][mask].detach().cpu().numpy().astype(int)
            cls = labels_one - 1
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
            target_rows = [[category_to_zero[ann["category_id"]], ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]] for ann in anns]
            targets = np.asarray(target_rows, dtype=np.float32).reshape(-1, 5)
            y, best = match(boxes, cls, scores, targets)
            if len(boxes):
                transformed_h, transformed_w = state["image_sizes"][0]
                scale_x, scale_y = transformed_w / image_info["width"], transformed_h / image_info["height"]
                transformed = torch.from_numpy(boxes * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)).cuda()
                features: dict[str, torch.Tensor] = state["features"]
                # The finest FPN level keeps candidate-local evidence while avoiding detector re-execution.
                fmap = next(iter(features.values()))
                feature = roi_align(fmap, [transformed], output_size=1, spatial_scale=fmap.shape[-1] / transformed_w, aligned=True).flatten(1).cpu().numpy().astype(np.float16)
                width, height = image_info["width"], image_info["height"]
                wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
                geometry = np.column_stack((boxes / np.array([width, height, width, height]), wh / np.array([width, height]), (wh[:, 0] * wh[:, 1]) / (width * height))).astype(np.float32)
                ctx = context(boxes, scores, (width, height))
                records.extend((str(path), feature[i], float(scores[i]), int(cls[i]), geometry[i], ctx[i], float(y[i]), float(best[i]), boxes[i]) for i in range(len(boxes)))
            if number % 100 == 0:
                print(f"{number}/{len(coco.getImgIds())} candidates={len(records)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, image=np.asarray([item[0] for item in records]), hidden=np.asarray([item[1] for item in records]), score=np.asarray([item[2] for item in records], np.float32), cls=np.asarray([item[3] for item in records], np.int16), geometry=np.asarray([item[4] for item in records], np.float32), context=np.asarray([item[5] for item in records], np.float32), label=np.asarray([item[6] for item in records], np.float32), best_iou=np.asarray([item[7] for item in records], np.float32), box=np.asarray([item[8] for item in records], np.float32))
    meta = {"detector": args.detector, "checkpoint": args.checkpoint, "coco_json": args.coco_json, "images": len(coco.getImgIds()), "candidates": len(records), "tp": int(sum(item[6] for item in records)), "fp": int(sum(item[6] == 0 for item in records)), "matching": "class-aware one-to-one IoU=0.50", "evidence": "frozen finest-FPN ROI evidence", "min_size": args.min_size, "max_size": args.max_size}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
