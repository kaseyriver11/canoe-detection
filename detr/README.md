# Canoe Object Detection Instructions (2021)

Run the following to collect the annotations and images

```
python detr/src/collect_data.py
```

Run the following to split into train/validation and create COCO files

```
python detr/src/voc2coco.py
```


### Run the COLAB Notebook

Images will need to be placed on your google drive and provided to the COLAB notebook in order to follow along.

This notebook only shows 10 epochs. We will need to increase this number to get back to our original results using Tensorflow.
