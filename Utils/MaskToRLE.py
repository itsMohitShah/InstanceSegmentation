import numpy as np
import matplotlib.pyplot as plt

def binary_mask_to_rle(binary_mask):
    h, w = binary_mask.shape
    mask_flat = binary_mask.reshape((h * w,)).astype(np.uint8)
    
    if len(mask_flat) == 0:
        return {'size': [h, w], 'counts': []}
    
    # Calculate where the values change
    diff = np.diff(mask_flat, prepend=np.array([-1], dtype=np.uint8))
    change_indices = np.where(diff != 0)[0]
    change_indices = np.append(change_indices, len(mask_flat))
    
    # Compute run lengths
    run_lengths = [int(x) for x in np.diff(change_indices)]
    run_values = mask_flat[change_indices[:-1]]
    
    # Build counts
    counts = []
    if run_values[0] == 1:
        counts.append(0)
    
    current_count = run_lengths[0]
    counts.append(current_count)
    
    for i in range(1, len(run_lengths)):
        counts.append(run_lengths[i])

    rle = {'size': [h, w], 'counts': counts}
    area = h*w
    return rle, area

def rle_to_pixel(img_size, rle):
    img_area = img_size[0]*img_size[1]
    canvas = np.zeros(img_area, dtype=np.uint8)
    toggle = False
    pixelcount = 0
    for idx, rle_pixel in enumerate(rle['counts']):
        if toggle == False:
            canvas[pixelcount:pixelcount+rle_pixel]=0
        else:
            canvas[pixelcount:pixelcount+rle_pixel]=1
        pixelcount += rle_pixel
        toggle = not toggle
    canvas = canvas.reshape(img_size)
    return canvas
