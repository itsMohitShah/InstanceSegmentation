from Utils.FileWalker import FileWalker
import shutil
import numpy as np
from tqdm import tqdm
import os
import cv2
from Utils.Open3DPlotter import generate_Open3D_pointcloud
path = 'dl_challenge'
rgb_FW = FileWalker(path,'rgb.jpg')
mask_FW = FileWalker(path,'mask.npy')
bbox3d_FW = FileWalker(path,'bbox3d.npy')
pc_FW = FileWalker(path,'pc.npy')

def rename_images(rgb_FW):
    path_destination = 'Images'
    os.makedirs(path_destination,exist_ok=True)
    for idx, image_path in tqdm(enumerate(rgb_FW.roots),total=len(rgb_FW.roots)):
        mask = np.load(mask_FW.roots[idx])
        numofinstances = mask.shape[0]
        renamed_images_path = os.path.join(path_destination,f'ID{str(idx).zfill(3)}--{numofinstances}instances.png')
        image = cv2.imread(image_path)
        cv2.imwrite(renamed_images_path,image)

def rename_bbox3d(bbox3d_FW):
    path_destination = 'Bbox3D'
    os.makedirs(path_destination,exist_ok=True)
    for idx, bbox3d_path in tqdm(enumerate(bbox3d_FW.roots),total=len(bbox3d_FW.roots)):
        bbox3d = np.load(bbox3d_path)
        numofinstances = bbox3d.shape[0]
        renamed_bbox3d_path = os.path.join(path_destination,f'ID{str(idx).zfill(3)}--{numofinstances}instances.npy')
        np.save(renamed_bbox3d_path,bbox3d)

def rename_mask(mask_FW):
    """Rename the mask files"""
    path_destination = 'Mask'
    os.makedirs(path_destination,exist_ok=True)
    for idx, mask_path in tqdm(enumerate(mask_FW.roots),total=len(mask_FW.roots)):
        mask = np.load(mask_path)
        numofinstances = mask.shape[0]

        renamed_mask_path = os.path.join(path_destination,f'ID{str(idx).zfill(3)}--{numofinstances}instances.npy')
        np.save(renamed_mask_path,mask)


def convert_mask_to_perpointlabel(mask_FW):
    """Convert the mask files to per point label"""
    path_destination = 'PerPointLabel'
    os.makedirs(path_destination,exist_ok=True)
    for idx, mask_path in tqdm(enumerate(mask_FW.roots),total=len(mask_FW.roots)):
        mask = np.load(mask_path)
        numofinstances = mask.shape[0]
        mask = np.argmax(mask,axis=0)
        renamed_mask_path = os.path.join(path_destination,f'ID{str(idx).zfill(3)}--{numofinstances}instances.npy')
        np.save(renamed_mask_path,mask)

# rename_images(rgb_FW)
# rename_bbox3d(bbox3d_FW)
# rename_mask(mask_FW)
