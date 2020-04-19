import tensorflow as tf
import numpy as np

def process(image_decoded, 
            padding_crop=5,
            target_height=75,
            target_width=300):
    """ Transforms an image to the format expected by the neural net.

    Parameters:
        image: Tensor or Numpy array of shape [HEIGHT, WIDTH, 1] 
    """

    #original_height = image_decoded.shape[0]
    #original_width = image_decoded.shape[1]
    original_height = tf.cast(tf.shape(image_decoded)[0], tf.int64)
    original_width = tf.cast(tf.shape(image_decoded)[1], tf.int64)

    #invert colors
    image_decoded = (image_decoded-255) * -1

    #pixels painted (character pixels)
    indices_true = tf.where(image_decoded > 20)

    #Crop only the formula area
    min_x, min_y = (tf.reduce_min(indices_true[:,0])-padding_crop, 
                    tf.reduce_min(indices_true[:,1])-padding_crop)
    max_x, max_y = (tf.reduce_max(indices_true[:,0])+padding_crop, 
                    tf.reduce_max(indices_true[:,1])+padding_crop)



    min_x = tf.maximum(np.int64(0), min_x)
    min_y = tf.maximum(np.int64(0), min_y)
    max_x = tf.minimum(max_x, original_height)
    max_y = tf.minimum(max_y, original_width)

    centered_formula = image_decoded[min_x:max_x, min_y:max_y, :]

    # Width-Padding to the required aspect ratio
    image_decoded = tf.image.resize_with_pad(
        centered_formula, target_height, target_width, method='area',
        antialias=True
    )

    #Normalize the image
    image_decoded = (image_decoded / 255) - .5

    return image_decoded


def load_png(filepath):
    return tf.image.decode_png(tf.io.read_file(filepath), channels=1)
