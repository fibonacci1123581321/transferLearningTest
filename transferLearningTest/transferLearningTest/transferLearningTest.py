# Developers: Zlatni & Prva

# coding: utf-8

# In[1]:

import os
import glob

from itertools import groupby
from collections import defaultdict

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import optimizer as opt
import dnn_model as dm
import get_loss as gl
import prepare_data as pd

# Parameters:
data_dir = 'C:\\Users\\neman\\Documents\\GitRepos\\transferLearningTest\\transferLearningTest\\transferLearningTest\\stanford dogs\\images'

model_type = 'VGG'

model_path = 'vgg_16.ckpt'
batch_size = 32
num_workers = 4
drop_out = 0.5
weight_decay = 5e-4


# optimization and optimizer
loss_type_classification = 'sparse_softmax_cross_entropy'
learning_rate = 1e-3
num_epochs = 20
optimizer_type = 'GradientDescentOptimizer'

# variables to reinitialize and optimize
var_to_reinitialize = 'vgg_16/fc8'
var_to_optimize = ['vgg_16/fc8', 'vgg_16/fc7', 'vgg_16/fc6', 'vgg_16/conv5']

# Function to get list of image names and class names
def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    files_and_labels = []
    validation_files_and_labels = []
    for label in labels:
        for i, f in enumerate(os.listdir(os.path.join(directory, label))):
            if i % 5 == 0:
                validation_files_and_labels.append((os.path.join(directory, label, f), label))
            else:
                files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    val_filenames, val_labels = zip(*validation_files_and_labels)
    
    filenames = list(filenames)
    labels = list(labels)

    val_filenames = list(val_filenames)
    val_labels = list(val_labels)

    unique_labels = list(set(labels))

    label_to_int = {}
    with open('..\\label_map.txt', 'w') as f:
        for i, label in enumerate(unique_labels):
            f.write(label)
            f.write('\t')
            f.write(str(i))
            f.write('\n')
            label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    val_labels = [label_to_int[l] for l in val_labels]

    return filenames, labels, val_filenames, val_labels

# Helper function to check accuracy
def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


# Get the list of filenames and corresponding list of labels for training et validation
train_filenames, train_labels, val_filenames, val_labels = list_images(data_dir)
num_classes = len(set(train_labels))

print(num_classes)


# Build the graph using appropriate class from slim
graph = tf.Graph()
ckpt = tf.train.latest_checkpoint(os.getcwd())

with graph.as_default():
    if ckpt == None:
        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool, name="is_training")

        # Get data iterators
        images_train, labels_train, train_init_op, iterator = pd.get_training_data(train_filenames, train_labels)
        images_val, labels_val, val_init_op, _ = pd.get_validation_data(val_filenames, val_labels, iterator_val = iterator)

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one.
        # Each model has a different architecture, so the final layer will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.

        logits = dm.predict(model_type, images_train, num_classes, weight_decay, drop_out, True)

        # Specify where the model checkpoint is (pretrained weights).
        assert(os.path.isfile(model_path))

        # Restore layers that we want
        var_restore_init, var_reinitialize_init = dm.init_variables(model_path, var_to_reinitialize)

        # Run inference and compute loss
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels_train, name="correct_prediction")
        gl.add(labels_train, logits, loss_type_classification)
        loss = tf.losses.get_total_loss(name = "training_loss")

        optimizer = opt.get_optimizer(optimizer_type, learning_rate = learning_rate)
        training_opt, initializer_opt = opt.get_training_operations(loss, vars_to_optimize_names = var_to_optimize, tf_optimizer = optimizer)

        saver = tf.train.Saver()
        tf.get_default_graph().finalize()
    else:
        """
        Restore checkpoint graph from latest present checkpoint
        """
        saver = tf.train.import_meta_graph(ckpt + ".meta")

        optimizer = opt.get_optimizer(optimizer_type, learning_rate = learning_rate)
        loss = graph.get_operation_by_name("training_loss")
        training_opt, initializer_opt = opt.get_training_operations(loss, computation_graph = graph, vars_to_optimize_names = var_to_optimize, tf_optimizer = optimizer)

        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")

        is_training = graph.get_tensor_by_name("is_training:0")
        train_init_op = graph.get_operation_by_name("train_init_op")
        val_init_op = graph.get_operation_by_name("val_init_op")

"""
## Run training procedure
"""

with tf.Session(graph = graph) as sess:

    if ckpt == None:
        var_restore_init(sess)
        sess.run(var_reinitialize_init)
    else:
        saver.restore(sess, ckpt)

    tf.summary.FileWriter(logdir = "./events/", graph = graph)
    sess.run(initializer_opt)

    for epoch in range(num_epochs):
        sess.run(train_init_op)
        
        while True:
            try:
                sess.run(training_opt)

            except tf.errors.OutOfRangeError:
                break

        # Check accuracy on the train and val sets every epoch.
        train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
        val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
        save_path = saver.save(sess, ".\\vgg16_stanford_dogs", global_step=epoch)
        print('Train accuracy: %f' % train_acc)
        print('Val accuracy: %f\n' % val_acc)
        print('Checkpoint saved to path: %s' % save_path)