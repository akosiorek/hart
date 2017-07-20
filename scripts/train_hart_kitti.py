########################################################################################
# 
# Hierarchical Attentive Recurrent Tracking
# Copyright (C) 2017  Adam R. Kosiorek, Oxford Robotics Institute, University of Oxford
# email:   adamk@robots.ox.ac.uk
# webpage: http://ori.ox.ac.uk
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# 
########################################################################################

# coding: utf-8

# In[1]:

import _init_paths
import os

import numpy as np
import tensorflow as tf
import argparse

from hart.data import disp
from hart.data.kitti.tools import get_data
from hart.model import util
from hart.model.attention_ops import FixedStdAttention
from hart.model.eval_tools import log_norm, log_ratios, log_values, make_expr_logger
from hart.model.tracker import HierarchicalAttentiveRecurrentTracker as HART
from hart.model.nn import AlexNetModel, IsTrainingLayer
from hart.train_tools import TrainSchedule, minimize_clipped


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir',
                    help='Path to a folder with kitti images', default='../../data/kitti_tracking/small')

parser.add_argument('--label_dir',
                    help='Path to a folder with kitti label files', default='../../data/kitti_tracking/label_02')

parser.add_argument('--alexnet_dir',
                    help='Path to a folder with bvlc_alexnet.npy', default='checkpoints')


parser.add_argument('--checkpoint_dir',
                    help='Path to a folder in which checkpoints will be stored', default='checkpoints/kitti')


parser.add_argument('--run_name',
                    help='Name of this training run', default='hart')

args = parser.parse_args()


checkpoint_name = 'model.ckpt'

batch_size = 10
img_size = 187, 621, 3
crop_size = 56, 56, 3

learning_rate = .33e-5

rnn_units = 100
norm = 'batch'
keep_prob = .75
debug = True

schedule = TrainSchedule(
    start_timesteps=5,
    max_timesteps=30,
    min_save_interval=500,
    min_logging_interval=500,
    max_logging_interval=100000,
    log_every_n_epochs=3,
    start_threshold=.7,
    max_threshold_decrease=.05,
    min_epochs_per_timestep=13,
)

img_size, crop_size = [np.asarray(i) for i in (img_size, crop_size)]
keys = ['img', 'bbox', 'presence']

# In[3]:

util.set_random_seed(0)

train_store, train, test_store, test = get_data(args.img_dir, args.label_dir, .8, img_size[:2], schedule.n_timesteps,
                                                schedule.max_timesteps,
                                                batch_size, n_threads=3, in_memory=True, sample_objects=-100,
                                                truncated_threshold=1., occluded_threshold=1, reverse=True, mirror=True)

x, y, p = [train[k] for k in keys]
test_x, test_y, test_p = [test[k] for k in keys]

is_training = IsTrainingLayer()
builder = AlexNetModel(args.alexnet_dir, layer='conv3', n_out_feature_maps=5, upsample=False, normlayer=norm,
                       keep_prob=keep_prob, is_training=is_training)

model = HART(x, y[0], p[0], batch_size, crop_size, builder,
             rnn_units,
             bbox_gain=[-4.78, -1.8, -3., -1.8],
             zoneout_prob=(.05, .05),
             normalize_glimpse=True,
             attention_module=FixedStdAttention,
             debug=debug,
             transform_init_features=True,
             transform_init_state=True,
             dfn_readout=True,
             feature_shape=(14, 14),
             is_training=is_training)

# In[4]:

schedule.train_store = train_store
print 'Num Test Batches: {}'.format(test_store.n_batches_per_epoch)

# In[5]:

p_bool = tf.cast(p, tf.bool)
loss = model.loss(y, p_bool, loss_type='intersection_over_union', crossentropy_weight=0.,
                  att_loss_type='tight_intersection', att_weight=1.,
                  obj_mask_weight=1., l2_weight=1e-4, adaptive=True)

# In[6]:

lr_tensor = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)

opt = tf.train.RMSPropOptimizer(lr_tensor, momentum=0.9)
train_step, gvs = minimize_clipped(opt, loss.value, clip_value=.3, return_gvs=True, soft=True)

log_norm((gv[0] for gv in gvs), 'grad_norm')
_ = log_ratios(gvs, 'grad_ratio')
for g, v in gvs:
    tf.summary.histogram(g.name, g)

# In[7]:

saver = tf.train.Saver(max_to_keep=100)

# In[8]:

sess = util.get_session()

# In[9]:

sess.run(tf.global_variables_initializer())
schedule.sess = sess
schedule.n_iter = util.try_resume_from_dir(sess, saver, args.checkpoint_dir, args.run_name)

# In[10]:

log_dir = util.make_logdir(args.checkpoint_dir, args.run_name)
checkpoint_path = os.path.join(log_dir, checkpoint_name)
summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir, sess.graph)


# In[11]:

def make_logger(exprs, name, num_batches, data=None):
    if data is not None:
        data = {x: data[0], y: data[1], p: data[2]}

    return make_expr_logger(sess, writer, num_batches, exprs, name, data)


log_test = make_logger(loss, 'Test', test_store.n_batches_per_epoch, [test_x, test_y, test_p])
log_train = make_logger(loss, 'Train', train_store.n_batches_per_epoch)


def log_imgs(n_seqs=1, test=False):
    if test:
        fd = sess.run({x: test_x, y: test_y, p: test_p})
        prefix = 'test'
    else:
        fd = None
        prefix = 'train'

    for n in xrange(n_seqs):
        name = '{}_iter_{}_seq_{}.png'.format(prefix, train_itr, n)
        name = os.path.join(log_dir, name)
        outputs = sess.run([x, y, p, model.pred_bbox, model.att_pred_bbox, model.glimpse], fd)
        i, b, pres, pb, ab, g = [c[:, n] for c in outputs]
        l = pres.sum(0).max()
        i, b, pres, pb, ab, g = [c[:l, n] for c in outputs]
        boxes = {'gt': b, 'predicted': pb, 'att': ab}
        disp.tile(-i, g, boxes, fig_size=(2, 2), img_size=(1, 3), mode='vertical', save_as=name)


def log(model, schedule):
    model.test_mode(sess)
    train_loss = log_train(schedule.n_iter, schedule.n_batches_per_epoch // 4)
    test_loss = log_test(train_itr)
    log_imgs()
    log_imgs(test=True)
    model.train_mode(sess)
    log_values(writer, schedule.n_iter, 'n_timesteps', schedule.n_timesteps)
    return train_loss, test_loss


# In[ ]:

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# In[ ]:

# training #######################################################
model.train_mode(sess)
for train_itr in schedule:

    if schedule.should_log:
        train_loss, _ = log(model, schedule)
        summary = sess.run(summaries)
        writer.add_summary(summary, train_itr)

        schedule.report_score(train_loss['loss/iou'])
        print sess.run(model.cell.att_bias)

    sess.run(train_step)

    if schedule.should_save:
        saver.save(sess, checkpoint_path, global_step=train_itr)
