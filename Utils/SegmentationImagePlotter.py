import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def inspect_segmentationmask(mask_FW, rgb_FW,colors):
    dict_numofinstances = {}
    path_imgwithmask = 'ImageswithSegmentationMask'
    os.makedirs(path_imgwithmask,exist_ok=True)
    for idx, _ in tqdm(enumerate(mask_FW.id),total=len(mask_FW.id)):
        mask = np.load(mask_FW.roots[idx])
        rgb = plt.imread(rgb_FW.roots[idx])

        numofinstances = mask.shape[0]
        try:
            dict_numofinstances[numofinstances] += 1
        except:
            dict_numofinstances[numofinstances] = 1
        mask_img = np.zeros_like(rgb)

        for dim in range(mask.shape[0]):
            x = np.where(mask[dim]==True)
            mask_img[x] = colors[dim%len(colors)]

        plt.figure(figsize=(15,15))
        plt.subplot(2,1,1)
        plt.imshow(rgb)
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.imshow(mask_img)
        plt.text(40,40,f'Number of instances: {numofinstances}',color='white',fontsize=15, backgroundcolor='gray')
        plt.axis('off')
        plt.savefig(os.path.join(path_imgwithmask,f'ID{str(idx).zfill(3)}--RGBMask--{numofinstances}instances.png'), bbox_inches='tight')
        plt.close()
        
    # print(dict_numofinstances)
    bars = plt.bar(dict_numofinstances.keys(),dict_numofinstances.values(), edgecolor='white')
    plt.title("Distribution of Number of Instances")
    plt.xlabel("Number of instances in an image")
    plt.ylabel("Number of occurences")
    plt.bar_label(bars, fontsize=10, color='black')
    plt.savefig('Distribution_of_Number_of_Instances.png')
    print("Execution complete")