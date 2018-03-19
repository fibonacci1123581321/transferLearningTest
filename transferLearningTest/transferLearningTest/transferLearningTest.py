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


# Check tensorflow version
# 

# In[2]:

tf.__version__


# Notebook parameters:

# In[3]:

data_dir = 'C:\\Users\\neman\\Documents\\GitRepos\\transferLearningTest\\transferLearningTest\\transferLearningTest\\stanford dogs\\images'

model_path = 'resnet_v1_101.ckpt'
batch_size = 32
num_workers = 4
num_epochs1 = 10
num_epochs2 = 10
learning_rate1 = 1e-3
learning_rate2 = 1e-5
dropout_keep_prob = 0.5
weight_decay = 5e-4


# Vgg mean per pixel used for normalization:
# 

# In[4]:

VGG_MEAN = [123.68, 116.78, 103.94]


# Function to get list of image names and class names

# In[5]:

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
    with open('C:\\temp\\label_map.txt', 'w') as f:
        for i, label in enumerate(unique_labels):
            f.write(label)
            f.write('\t')
            f.write(str(i))
            f.write('\n')
            label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    val_labels = [label_to_int[l] for l in val_labels]

    return filenames, labels, val_filenames, val_labels

# Test it with val directory

# In[6]:

#list_images(val_dir)


# Helper function to check accuracy

# In[7]:

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
# 

# In[8]:

train_filenames, train_labels, val_filenames, val_labels = list_images(data_dir)

num_classes = len(set(train_labels))


# In[9]:

print(num_classes)


# Build the graph using vgg class from slim

# In[10]:

new_graph = tf.Graph()
checkpoint_graph = tf.Graph()

# Run the optimizer
# In[ ]:

ckpt = tf.train.latest_checkpoint(os.getcwd())

if ckpt == None:

    with new_graph.as_default():

        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
        def _parse_function(filename, label):
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

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function,
            num_threads=num_workers, output_buffer_size=batch_size)
        train_dataset = train_dataset.map(training_preprocess,
            num_threads=num_workers, output_buffer_size=batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(batch_size)

        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,
            num_threads=num_workers, output_buffer_size=batch_size)
        val_dataset = val_dataset.map(val_preprocess,
            num_threads=num_workers, output_buffer_size=batch_size)
        batched_val_dataset = val_dataset.batch(batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                            batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset, name="train_init_op")
        val_init_op = iterator.make_initializer(batched_val_dataset, name="val_init_op")

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool, name="is_training")

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        rnet101 = tf.contrib.slim.nets.resnet_v1
        with slim.arg_scope(rnet101.resnet_arg_scope(weight_decay=weight_decay)):
            logits, _ = rnet101.resnet_v1_101(images, num_classes=num_classes, is_training=is_training)
            logits = tf.squeeze(logits)

        # Specify where the model checkpoint is (pretrained weights).
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        #variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['resnet_v1_101/logits'])
        #variables_to_restore_full = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
        #init_fn_full = tf.contrib.framework.assign_from_checkpoint_fn(tf.train.latest_checkpoint('C:\\Users\\Bubash\\source\\repos\\TransferLearningVGG\\TransferLearningVGG'), variables_to_restore_full)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('resnet_v1_101/logits')
        fc8_init = tf.variables_initializer(fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.AdamOptimizer(learning_rate1)
        fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables, name="fc8_train_op")
        slt_name = fc8_optimizer.get_slot_names()
        optimizer_slots = [ fc8_optimizer.get_slot(var, name) for name in fc8_optimizer.get_slot_names() for var in fc8_variables]
        optimizer_slots.extend([fc8_optimizer._beta1_power, fc8_optimizer._beta2_power])
        adam_initializer = tf.variables_initializer(optimizer_slots)
        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(learning_rate2)
        full_train_op = full_optimizer.minimize(loss, name="full_train_op")

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels, name="correct_prediction")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        list_of_variables = tf.all_variables()
        uninitialized_vars = tf.report_uninitialized_variables(list_of_variables)

        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

    with tf.Session(graph=new_graph) as sess:

        init_fn(sess)        # load the pretrained weights
        sess.run(fc8_init)   # initialize the new fc8 layer
        sess.run(adam_initializer)

        # Update only the last layer for a few epochs.
        for epoch in range(num_epochs1):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            current_batch_num = 1
            while True:
                try:
                    print('Starting batch: %d' % current_batch_num)
                    _ = sess.run(fc8_train_op, {is_training: True})
                    current_batch_num = current_batch_num + 1
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            save_path = saver.save(sess, ".\\resnet_v1_101_stanford_dogs", global_step=epoch)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Checkpoint saved to path: %s' % save_path)


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs1))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            save_path = saver.save(sess, ".\\resnet_v1_101_stanford_dogs", global_step=(num_epochs1+epoch))
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Checkpoint saved to path: %s' % save_path)

else:

    with checkpoint_graph.as_default():

        saver = tf.train.import_meta_graph(ckpt + ".meta")

    with tf.Session(graph=checkpoint_graph) as sess:

        saver.restore(sess, checkpoint_path)

        train_init_op = checkpoint_graph.get_operation_by_name("train_init_op")
        val_init_op = checkpoint_graph.get_operation_by_name("val_init_op")

        fc8_train_op = checkpoint_graph.get_operation_by_name("fc8_train_op")
        full_train_op = checkpoint_graph.get_operation_by_name("full_train_op")

        correct_prediction = checkpoint_graph.get_tensor_by_name("correct_prediction:0")
        is_training = checkpoint_graph.get_tensor_by_name("is_training:0")

        val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
        print('Val accuracy on first call: %f\n' % val_acc)

        # Update only the last layer for a few epochs.
        for epoch in range(num_epochs1):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            current_batch_num = 1
            while True:
                try:
                    print('Starting batch: %d' % current_batch_num)
                    _ = sess.run(fc8_train_op, {is_training: True})
                    current_batch_num = current_batch_num + 1
                except tf.errors.OutOfRangeError:
                    break
  
            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            save_path = saver.save(sess, ".\\resnet_v1_101_stanford_dogs", global_step=epoch)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Checkpoint saved to path: %s' % save_path)


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs1))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            save_path = saver.save(sess, ".\\resnet_v1_101_stanford_dogs", global_step=(num_epochs1+epoch))
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
            print('Checkpoint saved to path: %s' % save_path)

# In[ ]:
