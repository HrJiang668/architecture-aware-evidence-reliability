"""Export detector-preserving YOLO candidates and internal visual evidence for QCC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


def resolve_images(data: dict, split: str) -> list[Path]:
    root = Path(data["path"]); value = Path(data[split]); value = value if value.is_absolute() else root / value
    return sorted(p for p in value.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def labels_for(image: Path) -> Path:
    parts=list(image.parts); parts[parts.index("images")]='labels'; return Path(*parts).with_suffix('.txt')


def gt(image: Path, width: int, height: int, scale: float, pad_x: float, pad_y: float) -> np.ndarray:
    rows=[]; p=labels_for(image)
    if p.exists():
        for line in p.read_text().splitlines():
            c,cx,cy,w,h=map(float,line.split()[:5]); rows.append([int(c),(cx-w/2)*width*scale+pad_x,(cy-h/2)*height*scale+pad_y,(cx+w/2)*width*scale+pad_x,(cy+h/2)*height*scale+pad_y])
    return np.asarray(rows,dtype=np.float32).reshape(-1,5)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes): return np.empty(0,np.float32)
    x1=np.maximum(box[0],boxes[:,0]); y1=np.maximum(box[1],boxes[:,1]);x2=np.minimum(box[2],boxes[:,2]);y2=np.minimum(box[3],boxes[:,3])
    inter=np.maximum(x2-x1,0)*np.maximum(y2-y1,0); a=max(0,box[2]-box[0])*max(0,box[3]-box[1]); b=np.maximum(boxes[:,2]-boxes[:,0],0)*np.maximum(boxes[:,3]-boxes[:,1],0)
    return inter/np.maximum(a+b-inter,1e-8)


def match(boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    labels=np.zeros(len(boxes),np.float32); best=np.zeros(len(boxes),np.float32); used=set()
    for i in np.argsort(-scores):
        idx=np.where(targets[:,0].astype(int)==int(classes[i]))[0] if len(targets) else np.empty(0,int)
        if len(idx):
            values=iou(boxes[i],targets[idx,1:]); j=int(idx[values.argmax()]);best[i]=values.max()
            if best[i]>=.5 and j not in used: labels[i]=1;used.add(j)
    return labels,best


def competition_context(boxes: np.ndarray, scores: np.ndarray, input_size: int) -> np.ndarray:
    """Eight detector-preserving local competition descriptors per candidate."""
    if not len(boxes):
        return np.empty((0, 8), dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    wh = np.maximum(boxes[:, 2:] - boxes[:, :2], 0); areas = wh[:, 0] * wh[:, 1]
    xx = np.maximum(0, np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0]))
    yy = np.maximum(0, np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1]))
    overlap = xx * yy / np.maximum(areas[:, None] + areas[None, :] - xx * yy, 1e-6)
    np.fill_diagonal(overlap, 0)
    nearest = np.sort(distances, axis=1)
    rival = scores[None, :].repeat(len(scores), axis=0); np.fill_diagonal(rival, 0)
    min_distance = nearest[:, 0] / input_size if len(boxes) > 1 else np.ones(len(boxes))
    mean3 = nearest[:, :min(3, len(boxes) - 1)].mean(1) / input_size if len(boxes) > 1 else np.ones(len(boxes))
    mean8 = nearest[:, :min(8, len(boxes) - 1)].mean(1) / input_size if len(boxes) > 1 else np.ones(len(boxes))
    return np.column_stack((min_distance, mean3, mean8, overlap.max(1), overlap.mean(1), rival.max(1), scores - rival.max(1), (distances < input_size / 4).sum(1) / 8)).astype(np.float32)


def sample(feature: torch.Tensor, boxes: torch.Tensor, input_size: int) -> torch.Tensor:
    """ROI-average via grid sampling, preserving detector internal feature maps."""
    n=boxes.shape[0]; c,h,w=feature.shape[1:]
    if not n:return torch.empty((0,c),device=feature.device)
    cx=(boxes[:,0]+boxes[:,2])/2/input_size*2-1; cy=(boxes[:,1]+boxes[:,3])/2/input_size*2-1
    grid=torch.stack((cx,cy),-1).view(1,n,1,2)
    return F.grid_sample(feature,grid,align_corners=False).squeeze(0).squeeze(-1).T


def raw_candidates(prediction: torch.Tensor, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO's pre-NMS anchors without any ground-truth information."""
    raw = prediction[0].transpose(0, 1)
    cls_scores = raw[:, 4:]
    score, cls = cls_scores.max(1)
    keep = score > confidence
    # Keep the pre-NMS anchor order while supporting current PyTorch releases.
    original_indices = torch.nonzero(keep, as_tuple=False).flatten()
    raw = raw[keep]; score = score[keep]; cls = cls[keep]; cls_scores = cls_scores[keep]
    xywh = raw[:, :4]
    box = torch.column_stack((xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                              xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2))
    return box.detach().cpu().numpy(), score.detach().cpu().numpy(), cls.detach().cpu().numpy().astype(np.int16), cls_scores.detach().cpu().numpy(), original_indices.detach().cpu().numpy().astype(np.int64)


def suppression_mapping(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, selected: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Assign each pre-NMS candidate to the first same-class NMS survivor that removes it."""
    owner = np.full(len(boxes), -1, dtype=np.int32)
    ordered = selected[np.argsort(-scores[selected])]
    for survivor in ordered:
        free = np.flatnonzero((owner < 0) & (classes == classes[survivor]) & (scores <= scores[survivor]))
        if len(free):
            overlap = iou(boxes[survivor], boxes[free])
            owner[free[overlap > threshold]] = survivor
        owner[survivor] = survivor
    offsets = [0]; members = []
    for survivor in selected:
        suppressed = np.flatnonzero((owner == survivor) & (np.arange(len(owner)) != survivor))
        members.extend(suppressed.tolist()); offsets.append(len(members))
    return np.asarray(offsets, dtype=np.int64), np.asarray(members, dtype=np.int64)


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True);ap.add_argument('--data',required=True);ap.add_argument('--split',default='val');ap.add_argument('--out',required=True);ap.add_argument('--imgsz',type=int,default=640);ap.add_argument('--conf',type=float,default=.001);ap.add_argument('--iou',type=float,default=.7);ap.add_argument('--max-det',type=int,default=300)
    args=ap.parse_args(); data=yaml.safe_load(Path(args.data).read_text());images=resolve_images(data,args.split);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    model=YOLO(args.weights);net=model.model.cuda().eval(); evidence={}; hooks=[]
    # Initialize Ultralytics' preprocessing pipeline once; subsequent inference
    # uses the same forward pass for NMS, native features, and candidate links.
    model.predict(str(images[0]), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det, rect=False, device=0, verbose=False)
    # Dense YOLO variants expose their prediction-scale feature maps as inputs
    # to Detect, not as the Detect module's architecture-dependent output.
    detect = next(module for module in reversed(net.model) if module.__class__.__name__ == 'Detect')
    feature_indices = tuple(int(index) for index in detect.f)
    for idx in feature_indices: hooks.append(net.model[idx].register_forward_hook(lambda _m,_i,o,idx=idx:evidence.__setitem__(idx,o)))
    records=[];group_sizes=None;pre_boxes=[];pre_scores=[];pre_classes=[];pre_probs=[];pre_images=[];final_to_pre=[];suppressed_offsets=[0];suppressed_indices=[]
    try:
        for number,image in enumerate(images,1):
            # rect=False fixes a single, documented square input geometry for
            # both final candidates and the pre-NMS audit cache.
            bgr=cv2.imread(str(image)); tensor=model.predictor.preprocess([bgr]); raw_output=net(tensor)
            raw_tensor=raw_output[0] if isinstance(raw_output,(tuple,list)) else raw_output
            raw_box,raw_score,raw_cls,raw_prob,raw_indices=raw_candidates(raw_tensor,args.conf)
            nms_output, selected_tensor=non_max_suppression(
                raw_tensor, args.conf, args.iou, max_det=args.max_det, nc=0,
                end2end=getattr(net, 'end2end', False), return_idxs=True)
            selected_original=selected_tensor[0].detach().cpu().numpy().astype(np.int64)
            original_to_local=np.full(raw_tensor.shape[-1],-1,dtype=np.int64);original_to_local[raw_indices]=np.arange(len(raw_indices))
            selected=original_to_local[selected_original]
            if np.any(selected < 0):
                raise RuntimeError(f'pre-NMS index loss for {image}')
            # Use the detections returned by this exact NMS call.  This avoids
            # a second forward pass whose CUDA kernels may differ at a score
            # or suppression boundary.
            selected_detections=nms_output[0]
            original_boxes=selected_detections[:, :4].detach().cpu().numpy()
            scores=selected_detections[:, 4].detach().cpu().numpy()
            classes=selected_detections[:, 5].detach().cpu().numpy().astype(int)
            pre_start=len(pre_boxes);pre_boxes.extend(raw_box);pre_scores.extend(raw_score);pre_classes.extend(raw_cls);pre_probs.extend(raw_prob);pre_images.extend([str(image)]*len(raw_box))
            local_offsets,local_suppressed=suppression_mapping(raw_box,raw_score,raw_cls,selected,args.iou)
            height,width=bgr.shape[:2]; scale=min(args.imgsz/height,args.imgsz/width); pad_x=(args.imgsz-width*scale)/2;pad_y=(args.imgsz-height*scale)/2
            original_boxes[:,[0,2]]=(original_boxes[:,[0,2]]-pad_x)/scale
            original_boxes[:,[1,3]]=(original_boxes[:,[1,3]]-pad_y)/scale
            boxes=original_boxes.copy(); boxes[:,[0,2]]=boxes[:,[0,2]]*scale+pad_x; boxes[:,[1,3]]=boxes[:,[1,3]]*scale+pad_y
            if len(boxes):
                # Detection inference has just populated the hooks for this exact image.
                feats=[]; tb=torch.tensor(boxes,device='cuda',dtype=torch.float32)
                for idx in feature_indices: feats.append(sample(evidence[idx],tb,args.imgsz).detach().cpu().numpy())
                hidden=np.concatenate(feats,axis=1).astype(np.float16);group_sizes=np.asarray([feature.shape[1] for feature in feats],dtype=np.int32)
                targets=gt(image,width,height,scale,pad_x,pad_y);labels,best=match(boxes,classes,scores,targets)
                geometry=np.column_stack([boxes/args.imgsz,(boxes[:,2]-boxes[:,0])/args.imgsz,(boxes[:,3]-boxes[:,1])/args.imgsz,((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))/args.imgsz**2]).astype(np.float32)
                ctx=competition_context(boxes,scores,args.imgsz)
                for j in range(len(boxes)):
                    records.append((str(image),hidden[j],float(scores[j]),int(classes[j]),geometry[j],ctx[j],float(labels[j]),float(best[j]),boxes[j]))
                    final_to_pre.append(pre_start+int(selected[j]))
                    suppressed_indices.extend((pre_start+local_suppressed[local_offsets[j]:local_offsets[j+1]]).tolist());suppressed_offsets.append(len(suppressed_indices))
            if number%100==0: print(f'{number}/{len(images)} candidates={len(records)}',flush=True)
    finally:
        for hook in hooks:hook.remove()
    np.savez_compressed(out,image=np.array([x[0] for x in records]),hidden=np.array([x[1] for x in records]),hidden_group_sizes=group_sizes if group_sizes is not None else np.empty(0,np.int32),score=np.array([x[2] for x in records],np.float32),cls=np.array([x[3] for x in records],np.int16),geometry=np.array([x[4] for x in records],np.float32),context=np.array([x[5] for x in records],np.float32),label=np.array([x[6] for x in records],np.float32),best_iou=np.array([x[7] for x in records],np.float32),box=np.array([x[8] for x in records],np.float32),pre_nms_boxes=np.asarray(pre_boxes,np.float32).reshape(-1,4),pre_nms_scores=np.asarray(pre_scores,np.float32),pre_nms_classes=np.asarray(pre_classes,np.int16),pre_nms_class_probabilities=np.asarray(pre_probs,np.float16),pre_nms_images=np.asarray(pre_images),final_to_pre_nms=np.asarray(final_to_pre,np.int64),suppressed_offsets=np.asarray(suppressed_offsets,np.int64),suppressed_pre_nms_indices=np.asarray(suppressed_indices,np.int64))
    meta={'weights':args.weights,'data':args.data,'split':args.split,'images':len(images),'candidates':len(records),'tp':int(sum(x[6] for x in records)),'fp':int(sum(x[6]==0 for x in records)),'feature_indices':list(feature_indices),'matching':'class-aware one-to-one IoU=0.50','nms_iou':args.iou,'pre_nms_candidates':len(pre_boxes),'suppression_mapping':'same-class sequential NMS ownership at the detector NMS IoU threshold'}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
