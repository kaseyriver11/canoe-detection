import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(0, len(labels_str)+1))
    return dict(zip(labels_str, labels_ids))


def get_image_info(annotation_root, img_id: int):
    filename = annotation_root.findtext('filename')
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_dir: Path, label2id: Dict[str, int], json_name: str):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    img_id = 1
    print(f'Start converting {annotation_dir.name} for {json_name}!')
    for a_path in annotation_dir.glob("*.xml"):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root, img_id=img_id)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
        img_id += 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none',
                         'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(annotation_dir.parent.joinpath(json_name), 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():

    # Split the Images into training/testing
    train_dir = Path("detr/data/train/annotations")
    val_dir = Path("detr/data/val/annotations")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_dir = train_dir.parent
    val_dir = val_dir.parent

    random.seed(1111)
    for a_file in Path("detr/annotations").glob("*.xml"):
        name = a_file.name.replace(".xml", ".png")
        image_file = Path("detr/images", name)
        loc = "train"
        if random.random() > .7:
            loc = "val"
        # Copy Annotation
        shutil.copyfile(a_file, Path(
            f"detr/data/{loc}/annotations/", a_file.name))
        # Copy Image
        shutil.copyfile(image_file, Path(f"detr/data/{loc}/", image_file.name))

    label2id = get_label2id(labels_path="detr/labels.txt")
    convert_xmls_to_cocojson(
        annotation_dir=Path("detr/data/train/annotations"),
        label2id=label2id,
        json_name="custom_train.json"
    )
    convert_xmls_to_cocojson(
        annotation_dir=Path("detr/data/val/annotations"),
        label2id=label2id,
        json_name="custom_val.json"
    )


if __name__ == '__main__':
    main()
