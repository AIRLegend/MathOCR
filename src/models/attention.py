import tensorflow as tf

class SoftAttention(tf.keras.Model):
    """ Soft attention block based on the original implementation of
    https://arxiv.org/pdf/1609.04938v1.pdf
    """
    def __init__(self, nunits=256):
        super(SoftAttention, self).__init__()

        self.wc = tf.keras.layers.Dense(nunits, activation='linear', use_bias=False)
        self.wx =  tf.keras.layers.Dense(nunits, activation='linear', use_bias=False)
        self.beta = tf.Variable(tf.random.uniform((nunits,1), dtype=tf.float32))
    
    def call(self, context, features):
        hidd = self.wc(context)
        #feat = self.wx(features)
        feat=features

        si = tf.squeeze(tf.nn.tanh(hidd + feat)@self.beta, axis=-1)
        ai = tf.nn.softmax(si) #[B, L]

        z = ai*tf.reduce_sum(features, axis=-1) #[BATCH, C]
        
        return z
