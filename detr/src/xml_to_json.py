
from email.mime import image
import os
import glob
import xml.etree.ElementTree as ET
import xmltodict
import json
import numpy as np
from pathlib import Path
import shutil


def XML2JSON(xmlFiles, path_name):
    attrDict = dict()
    attrDict["categories"] = [
        {"supercategory": "none", "id": 0, "name": "canoe"}]
    images = list()
    annotations = list()
    image_id = 0
    annotation_id = 0
    for file in xmlFiles:
        annotation_path = file
        image = dict()
        doc = xmltodict.parse(
            open(annotation_path).read(), force_list=('object'))
        image['file_name'] = str(doc['annotation']['filename'])
        image['height'] = int(doc['annotation']['size']['height'])
        image['width'] = int(doc['annotation']['size']['width'])
        image['id'] = image_id
        print("File Name: {} and image_id {}".format(file, image_id))
        images.append(image)
        if 'object' in doc['annotation']:
            for obj in doc['annotation']['object']:
                for value in attrDict["categories"]:
                    annotation = dict()
                    if str(obj['name']) == value["name"]:
                        annotation["image_id"] = image_id
                        x1 = int(obj["bndbox"]["xmin"]) - 1
                        y1 = int(obj["bndbox"]["ymin"]) - 1
                        x2 = int(obj["bndbox"]["xmax"]) - x1
                        y2 = int(obj["bndbox"]["ymax"]) - y1
                        annotation["bbox"] = [x1, y1, x2, y2]
                        annotation["area"] = float(x2 * y2)
                        annotation["category_id"] = value["id"]
                        annotation["ignore"] = 0
                        annotation["iscrowd"] = 0
                        annotation["id"] = annotation_id
                        annotation["segmentation"] = [
                            [x1, y1, x1, (y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
                        annotation_id += 1
                        annotations.append(annotation)
                    else:
                        print("File: {} doesn't have any object".format(file))
        else:
            print("File: {} not found".format(file))
        image_id = image_id + 1

    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    jsonString = json.dumps(attrDict)
    with open(f"detr/data/{path_name}.json", "w") as f:
        f.write(jsonString)


if __name__ == "__main__":
    np.random.seed(1111)

    # Split the Images into training/testing
    train_dir = Path("detr/data/train/annotations")
    val_dir = Path("detr/data/val/annotations")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_dir = train_dir.parent
    val_dir = val_dir.parent

    for a_file in Path("detr/annotations").glob("*.xml"):
        image_file = Path(
            "detr/LakeVictoriaCoastline/outdir").joinpath(a_file.name.replace(".xml", ".jpeg"))
        loc = "train"
        if np.random.rand() > .7:
            loc = "val"
        # Copy Annotation
        shutil.copyfile(a_file, Path(
            f"detr/data/{loc}/annotations/").joinpath(a_file.name))
        # Copy Image
        shutil.copyfile(image_file,  Path(
            f"detr/data/{loc}/").joinpath(image_file.name))

    path = "detr/data/train/annotations"
    trainXMLFiles = glob.glob(os.path.join(path, '*.xml'))
    XML2JSON(trainXMLFiles, "custom_train2")

    path = "detr/data/val/annotations"
    valXMLFiles = glob.glob(os.path.join(path, '*.xml'))
    XML2JSON(valXMLFiles, "custom_val")

    # Test
    with open("/Users/krjones/Documents/krj/canoe-detection/detr/data/custom_val.json") as jf:
        val_json = json.load(jf)
