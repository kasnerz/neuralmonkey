#!/usr/bin/env python

"""

This is a traiing script for sequence to sequence learning.

"""

import sys
import os
import codecs
from shutil import copyfile

import tensorflow as tf

from utils import print_header, log
from config_loader import load_config_file
from learning_utils import training_loop
from dataset import Dataset

def get_from_configuration(configuration, name, expected_type=None, cond=None,
                           required=True, default=None):
    """ Checks whether a filed is in the configuration and returns it. """
    if name not in configuration:
        if required:
            raise Exception("Field \"{}\" is missing in the configuration.".format(name))
        else:
            return default

    value = configuration[name]
    if expected_type is not None and not isinstance(value, expected_type):
        raise Exception("Value of {} should be {}, but is {}.".format(name, expected_type,
                                                                      type(value)))
    if cond is not None and not cond(value):
        raise Exception("Value of {} does not satisfy conditions.".format(name))

    return value

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: train.py <ini_file>"
        exit(1)

    ini_file = sys.argv[1]
    log("Loading ini file: \"{}\"".format(ini_file), color='red')
    config_f = codecs.open(ini_file, 'r', 'utf-8')
    configuration = load_config_file(config_f)
    log("ini file loded.", color='red')


    name = get_from_configuration(configuration, 'name', basestring)

    print ""
    print_header(name)

    try:
        random_seed = get_from_configuration(configuration, 'random_seed', int, required=False)
        if random_seed is not None:
            tf.set_random_seed(random_seed)
        output = get_from_configuration(configuration, 'output', basestring)
        epochs = get_from_configuration(configuration, 'epochs', int, lambda x: x >= 0)
        trainer = get_from_configuration(configuration, 'trainer')
        encoders = get_from_configuration(configuration, 'encoders', list, lambda l: len(l) > 0)
        decoder = get_from_configuration(configuration, 'decoder')
        batch_size = get_from_configuration(configuration, 'batch_size', int, lambda x: x > 0)
        train_dataset = get_from_configuration(configuration, 'train_dataset', Dataset)
        val_dataset = get_from_configuration(configuration, 'val_dataset', Dataset)
        postprocess = get_from_configuration(configuration, 'postprocess')
        evaluation = get_from_configuration(configuration, 'evaluation', list)
        runner = get_from_configuration(configuration, 'runner')
        test_datasets = get_from_configuration(configuration, 'test_datasets', list,
                                               required=False, default=[])
    except Exception as e:
        log(e.message, color='red')
        exit(1)

    try:
        os.mkdir(output)
    except:
        log("Experiment directory \"{}\" already exists".format(output))
        exit(1)

    copyfile(ini_file, output+"/experiment.ini")
    os.system("git log -1 --format=%H > {}/git_commit".format(output))
    os.system("git --no-pager diff --color=always > {}/git_diff".format(output))

    log("Initializing the TensorFlow session.")
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=4,
                                            intra_op_parallelism_threads=4))
    sess.run(tf.initialize_all_variables())
    training_loop(sess, epochs, trainer, encoders + [decoder], decoder,
                  batch_size, train_dataset, val_dataset,
                  output, evaluation, runner, test_datasets=test_datasets)