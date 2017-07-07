# Hierarchical Attentive Object Tracking

This codebase (in progress) implements the system described in the paper:

Hierarchical Attentive Object Tracking

[Adam R. Kosiorek](https://www.linkedin.com/in/adamkosiorek/?locale=en_US), [Alex Bewley](http://ori.ox.ac.uk/mrg_people/alex-bewley/), [Ingmar Posner](http://ori.ox.ac.uk/mrg_people/ingmar-posner/)

See [the paper](https://arxiv.org/abs/1706.09262) for more details. Please contact Adam Kosiorek (adamk@robots.ox.ac.uk) if you have any questions.



# Training on KITTI
## Data preparation
    
    1. Download KITTI dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). We need [left color imagesi](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) and [tracking labels](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).
    2. Unpack data into a data folder; images should be in an image folder and labels should be in a label folder.
    3. Resize all the images to (621, 187).

## Training

    1. Download the AlexNet weights from [here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy) and put them file in the `checkpoints` folder.
    2. Run `python scripts/train_kitti.py --img_folder=path/to/image/folder --label_folder=/path/to/label/folder`. 

    The training script will save model checkpoints in the `checkpoints` folder and report train and test scores every couple of epochs. You can run tensorboard in the `checkpoints` folder to visualise training progress. Training should converge in about 400k iterations, which should take about 3 days. It might take a couple of hours between logging messages, so don't worry.
