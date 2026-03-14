import math
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=2):
    w, h, d = image.shape
    pad = [max(p - s, 0) for p, s in zip(patch_size, (w, h, d))]
    add_pad = any(pad)

    if add_pad:
        wl_pad, hl_pad, dl_pad = [p // 2 for p in pad]
        wr_pad, hr_pad, dr_pad = [p - l for p, l in zip(pad, (wl_pad, hl_pad, dl_pad))]
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode="constant")
    else:
        wl_pad = hl_pad = dl_pad = 0

    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                patch = torch.from_numpy(patch[None, None].astype(np.float32)).cuda()
                with torch.no_grad():
                    probs = torch.softmax(net(patch), dim=1).cpu().numpy()[0]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += probs
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    score_map /= np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map


def _cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        return np.array([metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)])
    return np.zeros(2)


def test_all_case(net, base_dir, test_list="val.txt", num_classes=2, patch_size=(96, 96, 96), stride_xy=64, stride_z=64):
    base_dir = Path(base_dir)
    with (base_dir / test_list).open("r", encoding="utf-8") as f:
        image_ids = [line.strip().split(",")[0] for line in f]

    total_metric = np.zeros((num_classes - 1, 2))
    for image_id in tqdm(image_ids, desc="Validation"):
        with h5py.File(base_dir / "data" / f"{image_id}.h5", "r") as h5f:
            image, label = h5f["image"][:], h5f["label"][:]
        prediction = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        for i in range(1, num_classes):
            total_metric[i - 1] += _cal_metric(label == i, prediction == i)
    return total_metric / len(image_ids)
