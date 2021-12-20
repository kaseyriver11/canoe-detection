# Canoe Object Detection Instructions (2021)

<<<<<<< HEAD
This repository contains the images and annotations needed for training. We do not provide instructions for obtaining and labeling these images. Images were taken from both Bing and Google Earth. We do not own these images.
=======
https://gist.github.com/kaseyriver11/6789e0c584cc555c8698be7dc95c4baf

This repository contains the images and labels needed for training. You will need to download the pretrained model from tensorflows model [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
>>>>>>> 126b7ccd4de4492d13a339a3049ee27c272857c3


### Environment Setup

If you already have python3.8 and pip installed on your computer, you can run the following from the repo directory:

Create your virtual environment:

```bash
pip install virtualenv
virtualenv python_env --python=python3.8
source python_env/bin/activate
pip install tensorflow
pip install jupyter
```

Clone the tensorflow object detection models repo:
```
git clone git@github.com:tensorflow/models.git
```

From here, we recomend following the official tensorflow documentation on setup found [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

Verify the installation:
```
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```


### Split the images train/test

```
python3 src/split_images.py 
```

### Convert XML Records to `.record`

```
python3 src/xml_to_tfrecord.py
```

### Download Pretrained Model

You will need to download the pretrained model from tensorflows model [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

We selected [this](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz) model.S

Unzip this file and place in `workspace/`

If you use the same modle, you can use the config file in the repo.

### Train

This takes awhile. Especially using only a CPU.

```bash
cd <root_of_repo>

python models/research/object_detection/model_main_tf2.py \
  --pipeline_config_path=workspace/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/pipeline.config \
  --model_dir=workspace/boat_model \
  --checkpoint_every_n=10_000 \
  --num_workers=2 \
  --alsologtostderr
```



You will need to update the config using the tutorial if you are not using the config we have already updated in the repo. Make sure batch size is lowered, or you wont have enough memory available to run. I used a batch size of 8.

## Train

### Split Images into Training/Testing

```
python src/split_images.py 
```

If you need the python file, from local singularity terminal:
`cp models/research/object_detection/model_main_tf2.py workspace/india-490/`

Open new jupyter terminal: 

```
cd /Tensorflow/workspace/india-490
python model_main_tf2.py --model_dir=models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 --pipeline_config_path=models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8/pipeline.config
```

#### To view progesss (this doesn't currently work):

New terminal:

```
cd /Tensorflow/models/research/object_detection
tensorboard --logdir=training/
```

Go [here](http://singularity.rtp.rti.org:6006/).


## Predict

Tutorial is [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_saved_model.html#sphx-glr-auto-examples-plot-object-detection-saved-model-py).

Use the provided jupyter [notebook](http://singularity.rtp.rti.org:4888/tree/Tensorflow/notebooks): `Detection-2.0.ipynb` and follow the outlined steps.

### Send to local

After predictions have been made, from `<repo>/pickles`:

```
scp  <user>@singularity.rtp.rti.org:/data/Projects/ethiopia_geo/workspace/india-490/folder1.pk .
scp  <user>@singularity.rtp.rti.org:/data/Projects/ethiopia_geo/workspace/india-490/folder2.pk .
scp  <user>@singularity.rtp.rti.org:/data/Projects/ethiopia_geo/workspace/india-490/folder3.pk .
```

### Create CSVs

From local, look at:

`python src/single_boxes.py`
