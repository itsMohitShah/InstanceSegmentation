from Utils.FileWalker import FileWalker
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pycocotools.mask as mask_utils
from skimage import measure
import json
from itertools import groupby
import cv2
from Utils.MaskToRLE import binary_mask_to_rle, rle_to_pixel
from Utils.COCOtemplate import export_COCO


def segmentationmask_to_polygon(mask_FW, rgb_FW):
    image_container = {}
    for idx, mask_name in tqdm(enumerate(mask_FW.namewithoutfiletype),total=len(mask_FW.id)):
        mask = np.load(mask_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])
        for individual_mask in mask:
            rle, area = binary_mask_to_rle(individual_mask)
            rows, cols = np.where(mask)
            if len(rows)>0 and len(cols)>0:
                x1 = np.min(cols)
                y1 = np.min(rows)
                x2 = np.max(cols)
                y2 = np.max(rows)
                bbox = [x1, y1, x2, y2]
            # mask_test = rle_to_pixel(rgb.shape[:2], rle)
            
def image_rle_bundler(mask_FW, rgb_FW, image_FW):
    master_dict = {}
    
    for idx, image_name in tqdm(enumerate(image_FW.namewithoutfiletype), total=len(image_FW.namewithoutfiletype)):
        mask = np.load(mask_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])
        image = plt.imread(image_FW.roots[idx])
        for individual_mask in mask:
            rle, area = binary_mask_to_rle(individual_mask)
            rows, cols = np.where(individual_mask)
            if len(rows)>0 and len(cols)>0:
                x1 = int(np.min(cols))
                y1 = int(np.min(rows))
                x2 = int(np.max(cols))
                y2 = int(np.max(rows))
                bbox = [x1, y1, x2, y2]
            # mask_test = rle_to_pixel(rgb.shape[:2], rle)
            # print(bbox)
            # cv2.rectangle(mask_test, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # plt.imshow(mask_test)
            # plt.show()
            if image_name not in master_dict.keys():
                master_dict[image_name] = {}
            master_dict[image_name]['image_height'] = image.shape[0]
            master_dict[image_name]['image_width'] = image.shape[1]
            if 'annotations' not in master_dict[image_name].keys():
                master_dict[image_name]['annotations'] = []
            master_dict[image_name]['annotations'].append({
                'segmentation': rle,
                'area': area,
                'bbox': bbox,
                'category_id': 0,
                'iscrowd': 0
            })

    return master_dict

if __name__ == '__main__':
    rgb_FW = FileWalker('dl_challenge','rgb.jpg')
    mask_FW = FileWalker('dl_challenge','mask.npy')
    image_FW = FileWalker('Images','.png')
    # segmentationmask_to_polygon(mask_FW, rgb_FW)
    json_path = 'COCO_Annotations.json'
    master_dict = image_rle_bundler(mask_FW, rgb_FW, image_FW)
    print("Preparing COCO JSON")
    COCO_dict = export_COCO(json_path, image_FW, master_dict)