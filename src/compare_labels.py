
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd


def main():
    r1_path = Path("google-images/Boxed_Images_Kasey")
    r2_path = Path("google-images/Boxed_Images")

    #
    r1_results = []
    for annotation_file in r1_path.glob("*.xml"):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        count = len(root.findall('object'))
        r1_results.append([annotation_file.name, count])
    r1_df = pd.DataFrame(r1_results, columns=['Name', 'Count'])

    r2_results = []
    for annotation_file in r2_path.glob("*.xml"):
        number = int(str(annotation_file.name).split("_")
                     [1].replace(".xml", ""))
        if number < 10000:
            continue
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        count = len(root.findall('object'))
        r2_results.append([annotation_file.name, count])
    r2_df = pd.DataFrame(r2_results, columns=['Name', 'Count'])

    r1 = r1_df.merge(r2_df, left_on='Name', right_on="Name", how='outer')

    r1[r1.Count_y > 0].Count_x.mean()
    r1[r1.Count_y > 0].Count_y.mean()

    # Average number of canoes in images labeled by reviewer 1, but not by reviewer 2
    r1[(r1.Count_x > 0) & r1.Count_y.isnull()].Count_x.mean()
    # Average number of canoes in images labeled by reviewer 2, but not by reviewer 1
    r1[(r1.Count_y > 0) & r1.Count_x.isnull()].Count_y.mean()

    final_results = []
    final_results.append(["Total Images", r1_df.shape[0], r2_df.shape[0]])
    final_results.append(
        ["Avg Labels", r1_df.Count.mean(), r2_df.Count.mean()])
    pd.DataFrame(final_results, columns=['Info', 'Labeler 1', 'Labeler 2'])

    # ----- Look at the final train/validation
    # Training
    train_results = []
    for annotation_file in Path("detr/data/train/annotations").glob("*.xml"):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        count = len(root.findall('object'))
        train_results.append([annotation_file.name, count])
    train_df = pd.DataFrame(train_results, columns=['Name', 'Count'])
    train_df.shape[0]
    train_df.Count.mean()
    train_df.Count.sum()

    # Validation
    val_results = []
    for annotation_file in Path("detr/data/val/annotations").glob("*.xml"):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        count = len(root.findall('object'))
        val_results.append([annotation_file.name, count])
    val_df = pd.DataFrame(val_results, columns=['Name', 'Count'])
    val_df.shape[0]
    val_df.Count.mean()
    val_df.Count.sum()


if __name__ == "__main__":
    main()
