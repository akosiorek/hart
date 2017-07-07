import os
import re
import random
import datetime

import tensorflow as tf
import numpy as np


def as_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def try_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_logdir(checkpoint_dir, run_name=None):
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")

    try_mkdir(checkpoint_dir)
    if run_name is not None:
        checkpoint_dir = os.path.join(checkpoint_dir, run_name)
        try_mkdir(checkpoint_dir)

    log_dir = os.path.join(checkpoint_dir, now)
    try_mkdir(log_dir)
    return log_dir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def extract_itr_from_modelfile(model_path):
    return int(model_path.split('-')[-1].split('.')[0])


def try_resume_from_dir(sess, saver, checkpoint_dir, run_name):
    run_folder = os.path.join(checkpoint_dir, run_name)
    if not os.path.exists(run_folder):
        return 0

    dirs = os.listdir(run_folder)
    dirs_as_date = []
    for d in dirs:
        try:
            date = datetime.datetime.strptime(d, '%Y_%m_%d_%H.%M')
            dirs_as_date.append((date, d))
        except:
            pass

    sorted_dirs = sorted(dirs_as_date, key=lambda x: x[0], reverse=True)
    sorted_dirs = [x[1] for x in sorted_dirs]

    model_files = []
    pattern = re.compile(r'.ckpt-[0-9]+$')

    for d in sorted_dirs:
        model_dir = os.path.join(run_folder, d)
        model_files = [f for f in os.listdir(model_dir) if pattern.search(f)]
        if model_files:
            break

    if model_files:
        model_file = max(model_files, key=extract_itr_from_modelfile)
        itr = extract_itr_from_modelfile(model_file)

        model_file = os.path.join(model_dir, model_file)
        print 'loading from', model_file
        saver.restore(sess, model_file)

        return itr
    else:
        print 'No modefile to resume from. Starting at iter = 0.'
        return 0


def get_session(allow_growth=True, mem_fraction=1.0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    config.gpu_options.per_process_gpu_memory_fraction = mem_fraction

    return tf.Session(config=config)