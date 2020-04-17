import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.pool1 = tf.keras.layers.MaxPool2D(2)
        
        self.conv2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.pool2 = tf.keras.layers.MaxPool2D(2)
        
        self.conv2h = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)

        self.conv3 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.pool3 = tf.keras.layers.MaxPool2D((2,2))

        self.conv4 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)
        self.pool4 = tf.keras.layers.MaxPool2D((2,2))
        
        self.conv5 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation=tf.nn.leaky_relu)

        self.encoded_res = tf.keras.layers.Reshape((4*18, 256))

     
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv2h(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.encoded_res(x)

        return x
