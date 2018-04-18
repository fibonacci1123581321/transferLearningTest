import os
import glob
import tensorflow as tf

from itertools import groupby
from collections import defaultdict

# Vgg mean per pixel used for normalization:
VGG_MEAN = [123.68, 116.78, 103.94]

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

# ----------------------------------------------------------------------
# DATASET CREATION using tf.contrib.data.Dataset
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

# The tf.contrib.data.Dataset framework uses queues in the background to feed in
# data to the model.
# We initialize the dataset with a list of filenames and labels, and then apply
# the preprocessing functions described above.
# Behind the scenes, queues will load the filenames, preprocess them with multiple
# threads and apply the preprocessing in parallel, and then batch the data
# Afterwards we define an iterator that can operator on either dataset.
# The iterator can be reinitialized by calling:
#     - sess.run(train_init_op) for 1 epoch on the training set
#     - sess.run(val_init_op)   for 1 epoch on the valiation set
# Once this is done, we don't need to feed any value for images and labels
# as they are automatically pulled out from the iterator queues.
# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `train_dataset`


def get_training_data(train_filenames, train_labels, iterator_train = None):
    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(parse_function,
        num_threads=8, output_buffer_size=32)
    
    train_dataset = train_dataset.map(training_preprocess,
        num_threads=8, output_buffer_size=32)
    
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    
    batched_train_dataset = train_dataset.batch(32)

    if iterator_train is None:
        iterator_train = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, 
                                                                 batched_train_dataset.output_shapes)

    images_train, labels_train = iterator_train.get_next()
    train_init_op = iterator_train.make_initializer(batched_train_dataset, name="train_init_op")

    return images_train, labels_train, train_init_op, iterator_train

def get_validation_data(val_filenames, val_labels, iterator_val = None):
   # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(parse_function,
        num_threads=8, output_buffer_size=32)

    val_dataset = val_dataset.map(val_preprocess,
        num_threads=8, output_buffer_size=32)
    
    batched_val_dataset = val_dataset.batch(32)

    if iterator_val is None:
        iterator_val = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types, 
                                                               batched_val_dataset.output_shapes)

    images_val, labels_val = iterator_val.get_next()
    val_init_op = iterator_val.make_initializer(batched_val_dataset, name="val_init_op")

    return images_val, labels_val, val_init_op, iterator_val