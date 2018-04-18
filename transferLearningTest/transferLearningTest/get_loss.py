import tensorflow as tf

# encapsulation of loss function adding to TensorFlow Graph
def add(labels, model_output, loss_type, weights = 1.0):
    # possible losses: absolute diff, computed weighted loss, cosine distance, hing loss, huber loss, log loss, 
    # mean pairwise squered error, mean squared error, sigmoid cross entropy, softmax cross entropy, sparse cross entropy

    # read the tensorflow doc for additional explinaiton for weight input param and other losses
    # https://www.tensorflow.org/api_docs/python/tf/losses

    if loss_type == 'compute_weighted_loss':
        return tf.losses.compute_weighted_loss(model_output, weights)
    else:
        return {
            'absolute_difference':tf.losses.absolute_difference,
            'cosine_distance':tf.losses.cosine_distance,
            'hinge_loss':tf.losses.hinge_loss,
            'huber_loss':tf.losses.huber_loss,
            'log_loss':tf.losses.log_loss,
            'mean_pairwise_squared_error':tf.losses.mean_pairwise_squared_error,
            'mean_squared_error':tf.losses.mean_squared_error,
            'sigmoid_cross_entropy':tf.losses.sigmoid_cross_entropy,
            'softmax_cross_entropy':tf.losses.softmax_cross_entropy,
            'sparse_softmax_cross_entropy':tf.losses.sparse_softmax_cross_entropy
            }[loss_type](labels, model_output, weights = weights)
