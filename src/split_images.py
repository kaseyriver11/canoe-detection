
import random
from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    # Set seed
    random.seed(1111)
    
    # Split the Images into training/testing
    image_dir = Path("images")
    train_dir = image_dir.joinpath("train/annotations")
    train_dir.mkdir(parents=True, exist_ok=True)
    train_dir = train_dir.parent
    test_dir = image_dir.joinpath("test/annotations")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_dir = test_dir.parent

    for file in image_dir.glob("*.jpg"):
        annotation_file = file.parent.joinpath(f"annotations/{file.name.replace('jpg', 'xml')}")
        # Test Set
        if random.random() < .3:
            temp_dir = test_dir
        else:
            temp_dir = train_dir
        copyfile(file, temp_dir.joinpath(file.name))
        copyfile(annotation_file, temp_dir.joinpath("annotations", annotation_file.name))
