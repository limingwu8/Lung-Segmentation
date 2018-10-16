import os
import numpy as np
import utils.array_tool as at
from utils.Config import opt
from imageio import imwrite

def compute_iou(pred_masks, gt_masks):
    pred_masks, gt_masks = np.squeeze(at.tonumpy(pred_masks)), np.squeeze(at.tonumpy(gt_masks))
    ious = []
    for i in range(len(pred_masks)):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i]

        union = np.sum(np.logical_or(pred_mask, gt_mask))
        intersection = np.sum(np.logical_and(pred_mask, gt_mask))
        iou = intersection/union
        ious.append(iou)
    batch_iou = np.sum(np.array(ious))

    return batch_iou


def save_pred_result(img_ids, images, pred_masks):
    for i in range(len(img_ids)):
        if not os.path.exists(os.path.join(opt.result_root, img_ids[i])):
            os.mkdir(os.path.join(opt.result_root, img_ids[i]))

        image = np.squeeze(images[i]).astype(np.uint8)
        imwrite(os.path.join(opt.result_root, img_ids[i], 'image.png'), image)

        pred_mask = np.squeeze(pred_masks[i])
        imwrite(os.path.join(opt.result_root, img_ids[i], 'pred_mask.png'), pred_mask)

        combined = np.multiply(image, pred_mask)
        imwrite(os.path.join(opt.result_root, img_ids[i], 'combined.png'), combined)
