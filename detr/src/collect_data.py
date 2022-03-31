
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def main():
    """The goal of this function is to collect annotations and images.
    """
    annotation_path = Path("detr/annotations/")
    annotation_path.mkdir(exist_ok=True)

    # Copy all rectlabel annotations
    for src_path in Path("google-images/annotations").glob("*.xml"):
        dst_path = annotation_path.joinpath(src_path.name)
        shutil.copy(src_path, dst_path)

    # Update and copy all other annotations
    for src_path in Path("google-images/Boxed_Images_David").glob("*.xml"):
        dst_path = annotation_path.joinpath(src_path.name)
        tree = ET.parse(src_path)
        # Add png
        root = tree.getroot()
        root.find('filename').text = root.find('filename').text + ".png"
        # Switch all "canoes" to "canoe"
        for member in root.findall('object'):
            member.find("name").text = "canoe"
            for item in ["pose", "truncated"]:
                child = member.find(item)
                member.remove(child)
        # Remove extra information
        for item in ["folder", "path", "segmented", "source"]:
            child = root.find(item)
            root.remove(child)

        with open(str(dst_path), 'wb') as f:
            tree.write(f)

    # Now - for every annotation - we also need an image
    annotation_path = Path("detr/annotations")
    new_image_path = Path("detr/images")
    new_image_path.mkdir(exist_ok=True)
    for annotation_file in annotation_path.glob("*.xml"):
        name = annotation_file.name.replace(".xml", ".png")
        number = int(str(name).replace(
            "LakeVictoria_", "").replace(".png", ""))
        if number < 5000:
            folder = "LakeVictoria_0-4999"
        elif number < 10000:
            folder = "LakeVictoria_5000-9999"
        elif number < 15000:
            folder = "LakeVictoria_10000-14999"
        src_path = str(Path("google-images").joinpath(folder, name))
        dst_path = str(new_image_path.joinpath(name))
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    main()
