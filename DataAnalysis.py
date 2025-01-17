import os
import numpy as np
import matplotlib.pyplot as plt
from Utils.FileWalker import FileWalker
from tqdm import tqdm
colors = [
    (255, 0, 0),        # Red
    (0, 255, 0),        # Green
    (0, 0, 255),        # Blue
    (255, 255, 0),      # Yellow
    (246, 109, 155),    # Magenta
    (0, 255, 255),      # Cyan
    (128, 0, 0),        # Maroon
    (0, 128, 0),        # Olive
    (149, 97, 226),     # Purple
    (229, 199, 218),    # Pink
    (229, 117, 4),      # Orange
    (175, 52, 45),      # Brown
    (255, 255, 255)     # White
]
mask_FW = FileWalker('dl_challenge','mask.npy')
bbox3d_FW = FileWalker('dl_challenge','bbox3d.npy')
pc_FW = FileWalker('dl_challenge','pc.npy')
rgb_FW = FileWalker('dl_challenge','rgb.jpg')

path_imgwithmask = 'ImageswithSegmentationMask'
os.makedirs(path_imgwithmask,exist_ok=True)

dict_numofinstances = {}

for idx, mask in tqdm(enumerate(mask_FW.id),total=len(mask_FW.id)):
    mask = np.load(mask_FW.roots[idx])
    bbox3d = np.load(bbox3d_FW.roots[idx])
    pc = np.load(pc_FW.roots[idx])
    rgb = plt.imread(rgb_FW.roots[idx])

    print(mask.shape)
    print(bbox3d.shape)
    print(pc.shape)
    print(rgb.shape)
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
    plt.savefig(os.path.join(path_imgwithmask,f'rgbmask_{idx}--{numofinstances}instances.png'), bbox_inches='tight')
    plt.close()
    
print(dict_numofinstances)
bars = plt.bar(dict_numofinstances.keys(),dict_numofinstances.values(), edgecolor='white')
plt.title("Distribution of Number of Instances")
plt.xlabel("Number of instances in an image")
plt.ylabel("Number of occurences")
plt.bar_label(bars, fontsize=10, color='black')
plt.savefig('Distribution_of_Number_of_Instances.png')
print("Execution complete")

