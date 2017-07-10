# Hierarchical Attentive Object Tracking

This is an official Tensorflow implementation of single object tracking in videos by using recurrent attentive recurrent neural networks, as presented in the following paper:

[A. R. Kosiorek](https://www.linkedin.com/in/adamkosiorek/?locale=en_US), [A. Bewley](http://ori.ox.ac.uk/mrg_people/alex-bewley/), [I. Posner](http://ori.ox.ac.uk/mrg_people/ingmar-posner/), ["Hierarchical Attentive Object Tracking", arXiv preprint, arxiv:1706.09262](https://arxiv.org/abs/1706.09262).

* **author**: Adam Kosiorek, Oxford Robotics Institue, University of Oxford
* **email**: adamk(at)robots.ox.ac.uk
* **paper**: https://arxiv.org/abs/1706.09262
* **webpage**: http://ori.ox.ac.uk/

## Installation
Install [Tensorflow v1.1](https://www.tensorflow.org/versions/r1.1/install/) and the following dependencies
 (using `pip install -r requirements.txt` (preferred) or `pip install [package]`):
* matplotlib==1.5.3
* numpy==1.12.1
* pandas==0.18.1
* scipy==0.18.1

## Demo
The notebook `scripts/demo.ipynb` contains a demo, which shows how to evaluate tracker on an arbitrary image sequence. By default, it runs on images located in `imgs` folder.
Before running the demo please download AlexNet weights first (described in the Training section).


## Data
    
1. Download KITTI dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). We need [left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and [tracking labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).
2. Unpack data into a data folder; images should be in an image folder and labels should be in a label folder.
3. Resize all the images to `(heigh=187, width=621)` e.g. by using the `scripts/resize_imgs.sh` script.

## Training

1. Download the AlexNet weights:
    * Execute `scripts/download_alexnet.sh` or
    * Download the weights from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put the file in the `checkpoints` folder.
2. Run

        python scripts/train_kitti.py --img_folder=path/to/image/folder --label_folder=/path/to/label/folder

The training script will save model checkpoints in the `checkpoints` folder and report train and test scores every couple of epochs. You can run tensorboard in the `checkpoints` folder to visualise training progress. Training should converge in about 400k iterations, which should take about 3 days. It might take a couple of hours between logging messages, so don't worry.

## Evaluation on KITTI dataset
The `scripts/eval_kitti.ipynb` notebook contains the code necessary to prepare (IoU, timesteps) curves for train and validation set of KITTI. Before running the evaluation:
* Download AlexNet weights (described in the Training section).
* Update image and label folder paths in the notebook.


## Release Notes
**Version 1.0**
* Original version from the paper. It contains the KITTI tracking experiment.