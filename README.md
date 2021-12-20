# Canoe Object Detection Instructions (2021)


This repository contains the images and labels needed for training. Images were taken from both Google Earth and Bing. We do not own these images. Labeling was completing using RectLabel. However, we now use LabelBox to perform labeling.

### Environment Setup

A python environment is only needed if you want to rerun the few python scripts in this repo. They are only used to split images into training/testing and to create tfrecord or json files. `tfrecord` files were originally used to train a tensorflow object detection model. `.json` files have been used recently to explore Facebook's [detr](https://github.com/facebookresearch/detr) model.

If you already have python3.8 and pip installed on your computer, you can run the following from the repo directory:

Create your virtual environment:

```bash
pip install virtualenv
virtualenv python_env --python=python3.8
source python_env/bin/activate
```

### Split the images train/test

```
python3 src/split_images.py 
```

### Convert XML Records to `.json`

```
python3 src/xml_to_json.py
```

### Run the COLAB Notebook

Fine tuning the DETR model can be done using this notebook:

https://gist.github.com/kaseyriver11/6789e0c584cc555c8698be7dc95c4baf

Images will nee to be placed on your google drive or uploaded to the COLAB notebook in order to follow along.
