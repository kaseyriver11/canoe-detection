# Canoe Object Detection Instructions (2021)

https://gist.github.com/kaseyriver11/6789e0c584cc555c8698be7dc95c4baf

This repository contains the images and labels needed for training. You will need to download the pretrained model from tensorflows model [zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

We found [this](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api) article to be helpful when switching from tensorflow 1.* to 2.*

### Environment Setup

If you already have python3.8 and pip installed on your computer, you can run the following from the repo directory:

Create your virtual environment:

```bash
pip install virtualenv
virtualenv python_env --python=python3.8
source python_env/bin/activate
pip install tensorflow==2.6.1
pip install jupyter
git clone git@github.com:tensorflow/models.git
```

You will need [protobuf](https://stackoverflow.com/questions/21775151/installing-google-protocol-buffers-on-mac) if you don't already have it.

For mac:
```
brew install protobuf
```

```
cd models/research
protoc models/research/object_detection/protos/*.proto --python_out=.
```

Setup tensorflow object detection:

```
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### Train

This takes awhile. Especially using only a CPU.

```bash
cd <root_of_repo>

python models/research/object_detection/model_main_tf2.py \
  --pipeline_config_path=workspace/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/pipeline.config \
  --model_dir=workspace/boat_model \
  --checkpoint_every_n=100 \
  --num_workers=2 \
  --alsologtostderr
```




### Download pretrained model:

Go to the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and select a model you want to use. We picked one based on speed and accuracy.

```
scp -r <...>/Downloads/efficientdet_d3_coco17_tpu-32 <user>@singularity:/data/Projects/ethiopia_geo/workspace/india-490/pre-trained-models
```

You will need to update the config using the tutorial if you are not using the config we have already updated in the repo. Make sure batch size is lowered, or you wont have enough memory available to run. I used a batch size of 8.

## Train

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
