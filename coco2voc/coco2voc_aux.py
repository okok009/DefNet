# from https://github.com/alicranck/coco2voc/tree/master
from typing import Sequence
import cv2
import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

VOID_PIXEL = 255
BORDER_THICKNESS = 7


def annotations_to_seg(annotations: Sequence[dict], coco_instance: COCO, apply_border: bool = False):
    image_details = coco_instance.loadImgs(annotations[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']

    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, annotations = annotations_to_mask(annotations, h, w)

    for i, mask in enumerate(masks):
        class_seg = np.where(class_seg > 0, class_seg, mask * annotations[i]['category_id'])
        # instance_seg = np.where(instance_seg > 0, instance_seg, mask * (i+1))
        # id_seg = np.where(id_seg > 0, id_seg, mask * annotations[i]['id'])

        if apply_border:    #看要不要畫輪廓
            border = get_border(mask, BORDER_THICKNESS)
            for seg in [class_seg, instance_seg, id_seg]:
                seg[border > 0] = VOID_PIXEL

    # return class_seg, instance_seg, id_seg.astype(np.int64)
    return class_seg


def annotation_to_rle(ann: dict, h: int, w: int):
    # ann中的segmentation為二維list，因為一個物體可能會被其他物件遮擋住，所以需要多個封閉曲線才能標註，各封閉曲線中的資料形式為[x1, y1, x2, y2, ...]。
    segm = ann['segmentation']
    if type(segm) == list:
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif type(segm['counts']) == list:
        rle = mask_utils.frPyObjects(segm, h, w)  # Uncompressed RLE
    else:
        rle = ann['segmentation']  # RLE
    return rle


def annotations_to_mask(annotations: Sequence[dict], h: int, w: int):
    masks = []
    # Smaller items first, so they won't be covered by overlapping segmentations
    annotations = sorted(annotations, key=lambda x: x['area'])
    for ann in annotations:
        rle = annotation_to_rle(ann, h, w)
        m = mask_utils.decode(rle)  #mask的值已確認0, 1
        masks.append(m)
    return masks, annotations


def get_border(mask: np.ndarray, thickness_factor: int = 7) -> np.ndarray:

    pil_mask = Image.fromarray(mask)  # Use PIL to reduce dependencies
    dilated_pil_mask = pil_mask.filter(ImageFilter.MaxFilter(thickness_factor))

    border = np.array(dilated_pil_mask) - mask

    return border
