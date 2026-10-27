"""Export frozen RF-DETR query trajectories for the QCC reliability head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from rfdetr import RFDETRSmall


def resolve_images(data: dict, split: str) -> list[Path]:
    root = Path(data["path"]); value = Path(data[split]); value = value if value.is_absolute() else root / value
    return sorted(path for path in value.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def labels_for(image: Path) -> Path:
    parts = list(image.parts); parts[parts.index("images")] = "labels"; return Path(*parts).with_suffix(".txt")


def coco_index(annotation: Path) -> dict:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload.get("images", [])}
    indexed: dict[str, tuple[int, int, list[tuple[int, float, float, float, float]]]] = {}
    for row in images.values():
        indexed[Path(row["file_name"]).name] = (int(row["width"]), int(row["height"]), [])
    for row in payload.get("annotations", []):
        image = images.get(int(row["image_id"]))
        if image is None or Path(image["file_name"]).name not in indexed:
            continue
        x, y, width, height = map(float, row["bbox"][:4])
        indexed[Path(image["file_name"]).name][2].append((int(row["category_id"]), x, y, width, height))
    return indexed


def ground_truth(image: Path, size: int, coco: dict | None = None) -> np.ndarray:
    if coco is not None and image.name in coco:
        width, height, annotations = coco[image.name]
        scale_x, scale_y = size / max(width, 1), size / max(height, 1)
        return np.asarray([(cls, x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y) for cls, x, y, bw, bh in annotations], dtype=np.float32).reshape(-1, 5)
    rows = []
    label = labels_for(image)
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            cls, cx, cy, width, height = map(float, line.split()[:5])
            rows.append((int(cls), (cx - width / 2) * size, (cy - height / 2) * size, (cx + width / 2) * size, (cy + height / 2) * size))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes): return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1]); x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0); area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    other = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return inter / np.maximum(area + other - inter, 1e-8)


def match(boxes: np.ndarray, cls: np.ndarray, score: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label = np.zeros(len(boxes), np.float32); best = np.zeros(len(boxes), np.float32); used: set[int] = set()
    for idx in np.argsort(-score):
        same = np.where(targets[:, 0].astype(int) == int(cls[idx]))[0] if len(targets) else np.empty(0, dtype=int)
        if len(same):
            values = iou(boxes[idx], targets[same, 1:]); winner = int(same[values.argmax()]); best[idx] = values.max()
            if best[idx] >= .5 and winner not in used: label[idx] = 1.; used.add(winner)
    return label, best


def competition(boxes: np.ndarray, score: np.ndarray, size: int) -> np.ndarray:
    if not len(boxes): return np.empty((0, 8), np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2; distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1); np.fill_diagonal(distance, np.inf)
    nearest = np.sort(distance, axis=1); wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0); area = wh[:, 0] * wh[:, 1]
    xx = np.maximum(0, np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])); yy = np.maximum(0, np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]))
    overlap = xx * yy / np.maximum(area[:, None] + area[None, :] - xx * yy, 1e-6); np.fill_diagonal(overlap, 0)
    rival = score[None, :].repeat(len(score), axis=0); np.fill_diagonal(rival, 0)
    if len(boxes) == 1: minimum = mean3 = mean8 = np.ones(1)
    else: minimum, mean3, mean8 = nearest[:, 0] / size, nearest[:, :min(3, len(boxes) - 1)].mean(1) / size, nearest[:, :min(8, len(boxes) - 1)].mean(1) / size
    return np.column_stack((minimum, mean3, mean8, overlap.max(1), overlap.mean(1), rival.max(1), score - rival.max(1), (distance < size / 4).sum(1) / 8)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--pretrain", required=True); parser.add_argument("--data"); parser.add_argument("--dataset-dir"); parser.add_argument("--split", default="val"); parser.add_argument("--out", required=True); parser.add_argument("--imgsz", type=int, default=512); parser.add_argument("--conf", type=float, default=.001); parser.add_argument("--max-candidates", type=int, default=100)
    args = parser.parse_args()
    coco = None
    if args.dataset_dir:
        root = Path(args.dataset_dir); split_dir = root / ("train" if args.split == "train" else "valid"); annotation = split_dir / "_annotations.coco.json"; payload = json.loads(annotation.read_text(encoding="utf-8")); categories = sorted(payload.get("categories", []), key=lambda row: int(row["id"])); data = {"path": str(root), args.split: split_dir.name, "names": [row["name"] for row in categories]}; coco = coco_index(annotation)
    else:
        if not args.data: raise ValueError("--data or --dataset-dir is required")
        data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8")); classes = len(data["names"])
    classes = len(data["names"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); model = RFDETRSmall(num_classes=classes, pretrain_weights=args.pretrain); net = model.model.model
    missing, unexpected = net.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected: raise RuntimeError(f"RF-DETR checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    net.cuda().eval(); states: dict[int, torch.Tensor] = {}; hooks = [layer.register_forward_hook(lambda _m, _i, output, index=index: states.__setitem__(index, output[0] if isinstance(output, (tuple, list)) else output)) for index, layer in enumerate(net.transformer.decoder.layers)]
    records = []; group_sizes = None
    try:
        for number, image in enumerate(resolve_images(data, args.split), 1):
            pixels = np.asarray(Image.open(image).convert("RGB")); tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().unsqueeze(0).cuda().div(255)
            tensor = F.interpolate(tensor, size=(args.imgsz, args.imgsz), mode="bilinear", align_corners=False); mean = tensor.new_tensor((.485, .456, .406)).view(1, 3, 1, 1); std = tensor.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
            states.clear()
            with torch.inference_mode(): output = net((tensor - mean) / std)
            logits = output["pred_logits"][0].float().sigmoid()[:, :classes]; score, cls = logits.max(1); keep = torch.where(score >= args.conf)[0]; keep = keep[torch.argsort(score[keep], descending=True)[:args.max_candidates]]
            if len(keep):
                raw = output["pred_boxes"][0, keep].float().cpu().numpy(); cx, cy, width, height = raw.T; boxes = np.stack(((cx - width / 2) * args.imgsz, (cy - height / 2) * args.imgsz, (cx + width / 2) * args.imgsz, (cy + height / 2) * args.imgsz), 1).clip(0, args.imgsz)
                score_np = score[keep].cpu().numpy().astype(np.float32); cls_np = cls[keep].cpu().numpy().astype(int); hidden = np.concatenate([states[index][0, keep].float().cpu().numpy() for index in sorted(states)], axis=1).astype(np.float16); group_sizes = np.asarray([states[index].shape[-1] for index in sorted(states)], dtype=np.int32)
                label, best = match(boxes, cls_np, score_np, ground_truth(image, args.imgsz, coco)); wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0); geometry = np.column_stack((boxes / args.imgsz, wh / args.imgsz, wh.prod(1) / args.imgsz ** 2)).astype(np.float32); ctx = competition(boxes, score_np, args.imgsz)
                records.extend((str(image), hidden[idx], float(score_np[idx]), int(cls_np[idx]), geometry[idx], ctx[idx], float(label[idx]), float(best[idx]), boxes[idx]) for idx in range(len(boxes)))
            if number % 100 == 0: print(f"{number} candidates={len(records)}", flush=True)
    finally:
        for hook in hooks: hook.remove()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, image=np.asarray([row[0] for row in records]), hidden=np.asarray([row[1] for row in records]), hidden_group_sizes=group_sizes if group_sizes is not None else np.empty(0,np.int32), score=np.asarray([row[2] for row in records], np.float32), cls=np.asarray([row[3] for row in records], np.int16), geometry=np.asarray([row[4] for row in records], np.float32), context=np.asarray([row[5] for row in records], np.float32), label=np.asarray([row[6] for row in records], np.float32), best_iou=np.asarray([row[7] for row in records], np.float32), box=np.asarray([row[8] for row in records], np.float32))
    meta = {"checkpoint": args.checkpoint, "pretrain": args.pretrain, "split": args.split, "images": len(resolve_images(data, args.split)), "candidates": len(records), "tp": int(sum(row[6] for row in records)), "fp": int(sum(row[6] == 0 for row in records)), "matching": "class-aware one-to-one IoU=0.50", "evidence": "frozen decoder-query trajectory plus competition context"}; out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8"); print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
