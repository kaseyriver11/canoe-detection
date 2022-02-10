# Canoe Object Detection Instructions (2021)


This repository contains the images and labels needed for training. Images were taken from both Google Earth and Bing. We do not own these images. Labeling was originally completing using RectLabel on a Mac. 

This project was originally completed using Tensorflow's object detection library. We have taken steps since then to convert the code to `torch` and `torchvision`. 

We provide the following Colab notebook to use for retraining and evaluation. We also found [this](https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae) tutorial to be helpful

### Run the COLAB Notebook

Images will need to be placed on your google drive and provided to the COLAB notebook in order to follow along.

This notebook only shows 10 epochs. We will need to increase this number to get back to our original results using Tensorflow.
