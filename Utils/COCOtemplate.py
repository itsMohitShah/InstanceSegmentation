import json
from tqdm import tqdm

def export_COCO(json_path, image_FW, master_dict):
    COCO_dict = {}
    # License
    COCO_dict['licenses'] = []
    dict_license = {}
    dict_license['id'] = 0
    dict_license['name'] = ''
    dict_license['url'] = ''
    COCO_dict['licenses'].append(dict_license)

    # Info
    COCO_dict['info'] = {}
    COCO_dict['info']['description'] = ''
    COCO_dict['info']['url'] = ''
    COCO_dict['info']['version'] = ''
    COCO_dict['info']['year'] = 2025
    COCO_dict['info']['contributor'] = ''
    COCO_dict['info']['date_created'] = ''

    # Categories
    COCO_dict['categories'] = []
    dict_category = {}
    dict_category['id'] = 0
    dict_category['name'] = 'object'
    dict_category['supercategory'] = 'object'
    COCO_dict['categories'].append(dict_category)

    # Images
    COCO_dict['images'] = []
    for idx, image_name in tqdm(enumerate(image_FW.namewithoutfiletype),total=len(image_FW.namewithoutfiletype)):
        dict_image = {}
        dict_image['id'] = idx
        dict_image['height'] = master_dict[image_name]['image_height']
        dict_image['width'] = master_dict[image_name]['image_width']
        dict_image['file_name'] = image_FW.names[idx]
        COCO_dict['images'].append(dict_image)

    # Annotations
    COCO_dict['annotations'] = []
    idx_box = 0
    for idx_image, namewithoutfiletype in enumerate(image_FW.namewithoutfiletype):
        for idx_annotation, annotation in enumerate(master_dict[namewithoutfiletype]['annotations']):
            dict_annotation = {}
            dict_annotation['id'] = idx_box
            dict_annotation['image_id'] = idx_image
            dict_annotation['category_id'] = 0
            dict_annotation['segmentation'] = annotation['segmentation']['counts']
            dict_annotation['area'] = float(annotation['area'])
            dict_annotation['bbox'] = annotation['bbox']
            dict_annotation['iscrowd'] = 0
            COCO_dict['annotations'].append(dict_annotation)
            idx_box += 1
    with open(json_path, 'w') as f:
        json.dump(COCO_dict, f)
    return COCO_dict
