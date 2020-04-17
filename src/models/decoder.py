import tensorflow as tf

class Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(Decoder, self).__init__()
    self.units = units

    self.embedding_dim=embedding_dim

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.attention = tf.keras.layers.Attention()
    self.lstm = tf.keras.layers.LSTM(self.units,
                                   return_sequences=False,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.fc_ot = tf.keras.layers.Dense(self.units, activation='tanh', use_bias=False)
    self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax', use_bias=True)

  def call(self, x, features, hidden, o_t_last):
    
    x = self.embedding(x)
    concated = tf.concat([tf.expand_dims(o_t_last, axis=1), x], axis=-1)
    ht, memory, cont = self.lstm(concated, initial_state=hidden)  #[Batch, self.units]
    memory = [memory, cont]

    ct = tf.squeeze(self.attention([tf.expand_dims(ht, axis=1), features]), axis=1)
    
    ot = self.fc_ot (
        tf.concat([ht, ct], axis=-1)
    )

    x = self.fc2(ot)

    return x, memory, ct, ot

  def reset_state(self, batch_size):
    return (tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units)))
