import tensorflow as tf


def acc_metric(real, pred, padding_char):
    real = tf.cast(real, dtype=tf.int32)
    pred = tf.cast(pred, dtype=tf.int32)
    
    mask_not_padding = tf.cast(tf.math.logical_not(
    			tf.math.equal(real[:,:pred.shape[1]], 
    							padding_char)), 
    dtype=tf.int32)
    
    equals =tf.cast(tf.math.equal(real[:,:pred.shape[1]]
                           , pred), dtype=tf.int32) * mask_not_padding
    
    acc = tf.reduce_sum(equals) / tf.reduce_sum(mask_not_padding)
    
    return acc
