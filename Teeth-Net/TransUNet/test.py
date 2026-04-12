# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# --- 可视化 / 张量工具 ---
import PIL.Image as PILImage
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2

# ================== 可视化与评估公用的后处理 ==================
POSTPROC = True  # 是否做后处理（连通域过滤 + 闭运算）
POST_MIN_AREA = 80  # 小连通域面积阈值（px）
POST_KERNEL = 3  # 闭运算核大小（奇数）
# ============================================================

# ================== 评估口径开关 ==================
EVAL_USE_POSTPROC = True  # 评估也应用与可视化一致的后处理
EVAL_PRESENT_ONLY = True  # 只对GT出现的前景类做宏平均
EVAL_STRICT_REPORT = False  # 同时报告strict口径（GT空但Pred非空→记0并计入平均）
EVAL_ROI = False  # 额外报告ROI（仅在GT外接框±pad内）
ROI_PAD = 5  # ROI扩展像素
IGNORE_INDEX = None  # 若标签有忽略值（如255）就写 255；否则 None
EPS = 1e-6
# =================================================

# 自定义调色板（类ID: RGB；0 背景不着色）
PALETTE = {
    1: (255, 0, 0),  # periapical - 红
    2: (0, 255, 0),  # caries - 绿
    3: (0, 0, 255),  # furcation - 蓝
    4: (255, 255, 0),  # impacted - 黄
}


def _mask_to_color(mask: np.ndarray, palette=PALETTE) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in palette.items():
        color[mask == cid] = rgb
    return color


def _make_rgb_from_tensor(img_tensor) -> np.ndarray:
    """image: [C,H,W] → [H,W,3] uint8；单通道用第1通道转灰度"""
    img = img_tensor.detach().cpu()
    if img.shape[0] >= 3:
        x = img[:3].clamp_(img.min(), img.max())
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
        return x
    g = img[0].numpy()
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    g = (g * 255.0).round().astype(np.uint8)
    return np.dstack([g, g, g])


def _blend_overlay(base_rgb: np.ndarray, color_mask: np.ndarray, alpha=0.5) -> np.ndarray:
    overlay = base_rgb.copy()
    m = (color_mask.sum(axis=2) > 0)
    overlay[m] = (base_rgb[m] * (1 - alpha) + color_mask[m] * alpha).astype(np.uint8)
    return overlay


def _postproc_mask(pred_2d, min_area=80, k=3):
    """简单后处理：去小连通域 + 闭运算"""
    out = pred_2d.copy()
    for cid in [1, 2, 3, 4]:
        m = (pred_2d == cid).astype(np.uint8)
        if m.sum() == 0:
            continue
        # 去小连通域
        n, lab = cv2.connectedComponents(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if (lab == i).sum() >= min_area:
                keep[lab == i] = 1
        # 闭运算连接小裂缝
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)
        out[(out == cid) & (keep == 0)] = 0
        out[keep == 1] = cid
    return out


def _apply_postproc_np(pred_np):
    if not EVAL_USE_POSTPROC:
        return pred_np
    return _postproc_mask(pred_np, POST_MIN_AREA, POST_KERNEL)


def _get_roi_from_gt(gt_np):
    if IGNORE_INDEX is not None:
        valid = (gt_np > 0) & (gt_np != IGNORE_INDEX)
    else:
        valid = (gt_np > 0)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - ROI_PAD), min(gt_np.shape[0], ys.max() + 1 + ROI_PAD)
    x0, x1 = max(0, xs.min() - ROI_PAD), min(gt_np.shape[1], xs.max() + 1 + ROI_PAD)
    return y0, y1, x0, x1


def _per_class_metrics(pred_np, gt_np, num_classes):

    per_dice = np.zeros(num_classes, dtype=np.float64)
    per_ppv = np.zeros(num_classes, dtype=np.float64)
    per_sens = np.zeros(num_classes, dtype=np.float64)
    present_ids, strict_ids = [], []
    for c in range(1, num_classes):
        g = (gt_np == c)
        p = (pred_np == c)
        TP = int((g & p).sum())
        FP = int((~g & p).sum())
        FN = int((g & ~p).sum())
        gt_has = TP + FN > 0
        pred_has = TP + FP > 0

        if gt_has:
            present_ids.append(c)
        if gt_has or pred_has:
            strict_ids.append(c)
        else:
            continue

        per_dice[c] = (2.0 * TP + EPS) / (2.0 * TP + FP + FN + EPS)
        per_ppv[c] = (TP + EPS) / (TP + FP + EPS)
        per_sens[c] = (TP + EPS) / (TP + FN + EPS)
    return per_dice, per_ppv, per_sens, present_ids, strict_ids


def _aggregate_mean(per_arr, ids):
    if len(ids) == 0:
        return 1.0  # 整张图无前景：定义为1或跳过；这里定义为1更稳
    return float(np.mean([per_arr[c] for c in ids]))



def save_png_pred(image, pred_eval_np, label, out_dir, case_name, alpha=0.5):

    os.makedirs(out_dir, exist_ok=True)

    img_rgb = _make_rgb_from_tensor(image[0])  # [H,W,3] uint8
    vutils.save_image(
        torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0,
        os.path.join(out_dir, f"{case_name}_img.png")
    )

    PILImage.fromarray(pred_eval_np).save(os.path.join(out_dir, f"{case_name}_pred.png"))

    lab0 = None
    if label is not None:
        lab0 = label[0].detach().cpu().numpy().astype(np.uint8)
        PILImage.fromarray(lab0).save(os.path.join(out_dir, f"{case_name}_gt.png"))

    # 叠加图
    overlay_pred = _blend_overlay(img_rgb, _mask_to_color(pred_eval_np), alpha)
    PILImage.fromarray(overlay_pred).save(os.path.join(out_dir, f"{case_name}_overlay.png"))
    if lab0 is not None:
        overlay_gt = _blend_overlay(img_rgb, _mask_to_color(lab0), alpha)
        PILImage.fromarray(overlay_gt).save(os.path.join(out_dir, f"{case_name}_overlay_gt.png"))


# ================== CLI ==================
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='../data/Synapse/test_vol_h5')
parser.add_argument('--dataset', type=str, default='Synapse')
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')

parser.add_argument('--max_iterations', type=int, default=20000)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--is_savenii', action="store_true")

parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')

parser.add_argument('--test_save_dir', type=str, default='../predictions')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vit_patches_size', type=int, default=16)

# PNG 输出目录
parser.add_argument('--png_out', type=str, default=r"D:\TransUnetOR\TransUNet1\pred_png1")

args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    sum_dice_present = np.zeros(args.num_classes);
    cnt_dice_present = np.zeros(args.num_classes, dtype=int)
    sum_ppv_present = np.zeros(args.num_classes);
    cnt_ppv_present = np.zeros(args.num_classes, dtype=int)
    sum_sen_present = np.zeros(args.num_classes);
    cnt_sen_present = np.zeros(args.num_classes, dtype=int)

    sum_dice_strict = np.zeros(args.num_classes);
    cnt_dice_strict = np.zeros(args.num_classes, dtype=int)
    sum_ppv_strict = np.zeros(args.num_classes);
    cnt_ppv_strict = np.zeros(args.num_classes, dtype=int)
    sum_sen_strict = np.zeros(args.num_classes);
    cnt_sen_strict = np.zeros(args.num_classes, dtype=int)

    roi_dice_list, roi_ppv_list, roi_sen_list = [], [], []

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label = sampled_batch["image"], sampled_batch["label"]  # [1,C,H,W], [1,H,W]
        case_name = sampled_batch['case_name'][0]
        _, _, H, W = image.size()

        metric_i = test_single_volume(
            image, label, model, classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing
        )
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f',
                     i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])

        with torch.no_grad():
            img_np = image.squeeze(0).cpu().numpy()  # [C,H,W]
            if (H, W) != (args.img_size, args.img_size):
                from scipy.ndimage import zoom
                img_resized = zoom(img_np, (1, args.img_size / H, args.img_size / W), order=3)
            else:
                img_resized = img_np
            inp = torch.from_numpy(img_resized).unsqueeze(0).float().cuda()  # [1,C,sz,sz]

            logits = model(inp)  # [1,C,sz,sz]
            logits_up = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            pred = torch.argmax(torch.softmax(logits_up, dim=1), dim=1, keepdim=True)  # [1,1,H,W]

        pred_np = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
        lab_np = label[0].detach().cpu().numpy().astype(np.uint8)

        pred_eval = _apply_postproc_np(pred_np)

        if IGNORE_INDEX is not None:
            m_valid = (lab_np != IGNORE_INDEX)
            pred_eval = np.where(m_valid, pred_eval, 0)
            lab_np = np.where(m_valid, lab_np, 0)

        per_dice, per_ppv, per_sens, present_ids, strict_ids = _per_class_metrics(
            pred_eval, lab_np, args.num_classes
        )
        md_present = _aggregate_mean(per_dice, present_ids)
        mp_present = _aggregate_mean(per_ppv, present_ids)
        ms_present = _aggregate_mean(per_sens, present_ids)

        md_strict = mp_strict = ms_strict = None
        if EVAL_STRICT_REPORT:
            md_strict = _aggregate_mean(per_dice, strict_ids)
            mp_strict = _aggregate_mean(per_ppv, strict_ids)
            ms_strict = _aggregate_mean(per_sens, strict_ids)

        md_roi = mp_roi = ms_roi = None
        if EVAL_ROI:
            roi = _get_roi_from_gt(lab_np)
            if roi is not None:
                y0, y1, x0, x1 = roi
                d_roi, p_roi, s_roi, pres_roi, _ = _per_class_metrics(
                    pred_eval[y0:y1, x0:x1], lab_np[y0:y1, x0:x1], args.num_classes
                )
                md_roi = _aggregate_mean(d_roi, pres_roi)
                mp_roi = _aggregate_mean(p_roi, pres_roi)
                ms_roi = _aggregate_mean(s_roi, pres_roi)
                roi_dice_list.append(md_roi);
                roi_ppv_list.append(mp_roi);
                roi_sen_list.append(ms_roi)

        logging.info(
            f"{case_name} "
            f"present-only: Dice={md_present:.4f} | PPV={mp_present:.4f} | Sens={ms_present:.4f}"
            + (
                f" || strict: Dice={md_strict:.4f} | PPV={mp_strict:.4f} | Sens={ms_strict:.4f}" if EVAL_STRICT_REPORT else "")
            + (f" || ROI: Dice={md_roi:.4f} | PPV={mp_roi:.4f} | Sens={ms_roi:.4f}" if md_roi is not None else "")
            + f" | present={present_ids}"
        )

        for c in present_ids:
            sum_dice_present[c] += per_dice[c];
            cnt_dice_present[c] += 1
            sum_ppv_present[c] += per_ppv[c];
            cnt_ppv_present[c] += 1
            sum_sen_present[c] += per_sens[c];
            cnt_sen_present[c] += 1
        if EVAL_STRICT_REPORT:
            for c in strict_ids:
                sum_dice_strict[c] += per_dice[c];
                cnt_dice_strict[c] += 1
                sum_ppv_strict[c] += per_ppv[c];
                cnt_ppv_strict[c] += 1
                sum_sen_strict[c] += per_sens[c];
                cnt_sen_strict[c] += 1

        save_png_pred(image, pred_eval, label, args.png_out, case_name)

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f', i, metric_list[i - 1][0], metric_list[i - 1][1])
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance (orig): mean_dice=%f mean_hd95=%f', performance, mean_hd95)

    def _safe_mean(sum_arr, cnt_arr):
        mean_arr = np.divide(sum_arr, np.maximum(cnt_arr, 1))
        vals = [mean_arr[c] for c in range(1, args.num_classes) if cnt_arr[c] > 0]
        return (np.mean(vals) if len(vals) > 0 else 1.0), mean_arr

    md_mean, md_cls = _safe_mean(sum_dice_present, cnt_dice_present)
    mp_mean, mp_cls = _safe_mean(sum_ppv_present, cnt_ppv_present)
    ms_mean, ms_cls = _safe_mean(sum_sen_present, cnt_sen_present)
    logging.info(f'Dataset (present-only): Dice={md_mean:.4f} | PPV={mp_mean:.4f} | Sens={ms_mean:.4f}')
    for c in range(1, args.num_classes):
        if cnt_dice_present[c] > 0:
            logging.info(f'  class {c}: Dice={md_cls[c]:.4f} PPV={mp_cls[c]:.4f} Sens={ms_cls[c]:.4f} '
                         f'(n={int(cnt_dice_present[c])})')
        else:
            logging.info(f'  class {c}: (no GT) skipped')

    if EVAL_STRICT_REPORT:
        md_mean_s, _ = _safe_mean(sum_dice_strict, cnt_dice_strict)
        mp_mean_s, _ = _safe_mean(sum_ppv_strict, cnt_ppv_strict)
        ms_mean_s, _ = _safe_mean(sum_sen_strict, cnt_sen_strict)
        logging.info(f'Dataset (strict): Dice={md_mean_s:.4f} | PPV={mp_mean_s:.4f} | Sens={ms_mean_s:.4f}')

    if EVAL_ROI and len(roi_dice_list) > 0:
        logging.info(f'Dataset ROI (present-only): '
                     f'Dice={np.mean(roi_dice_list):.4f} | PPV={np.mean(roi_ppv_list):.4f} | Sens={np.mean(roi_sen_list):.4f}')

    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        cudnn.benchmark, cudnn.deterministic = False, True
    random.seed(args.seed);
    np.random.seed(args.seed)
    torch.manual_seed(args.seed);
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 5,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    args.is_savenii = True
    args.test_save_dir = r"D:\TransUnetOR\TransUNet1\predictions"

    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = (r'D:\TransUnetOR\TransUNet1\model\TU_Synapse512\TU_pretrain_R50-ViT-B_16_skip3_epo120_bs12_lr0.005_512'
                r'\epoch_99.pth')
    net.load_state_dict(torch.load(snapshot, map_location="cuda"))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + '/' + snapshot_name + ".txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args));
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = r"D:\TransUnetOR\TransUNet1\predictions"
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    os.makedirs(args.png_out, exist_ok=True)

    inference(args, net, test_save_path)
