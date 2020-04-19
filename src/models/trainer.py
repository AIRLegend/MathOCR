import tensorflow as tf
import numpy as np
import time
import datetime

from .encoder import Encoder
from .decoder import Decoder

from utils.metrics import acc_metric

class Im2SeqTrainer():

        def __init__(self, 
                     encoder, 
                     decoder,
                     pad_id,
                     logdir='../../models/logs/', 
                     savedir='../../models/',
                     restore_checkpoint=False,
                     optimizer=None):

            self.logdir = logdir
            self.savedir = savedir
            self.pad_id = pad_id
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.train_log_dir = logdir + 'gradient_tape/' + current_time + '/train'
            self.test_log_dir = logdir + 'gradient_tape/' + current_time + '/test'
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

            if optimizer is None:
                self.optimizer = tf.keras.optimizers.Adam(lr=3e-4)
            else:
                self.optimizer = optimizer

            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False, reduction='none')

            self.encoder = encoder
            self.decoder = decoder

            checkpoint_path = self.savedir + "checkpoints/train"
            ckpt = tf.train.Checkpoint(encoder=self.encoder, 
                                       decoder=self.decoder,
                                       optimizer = self.optimizer) 
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                                           checkpoint_path, 
                                                           max_to_keep=2)
            if restore_checkpoint:
                self.ckpt_manager.restore(self.ckpt_manager.latest_checkpoint)
        
        @tf.function
        def loss_function(self, real, pred):
            mask_not_padding = tf.math.logical_not(tf.math.equal(real, self.pad_id))
            loss_ = self.loss_object(real, pred)
            mask_not_padding = tf.cast(mask_not_padding, tf.float32) * 5
            loss_ *= mask_not_padding
            
            return tf.reduce_mean(loss_)

        @tf.function
        def _train_step(self, batch_input, batch_target):
            loss = 0
            seq_len = batch_target.shape[1] - 1
            batch_size = batch_target.shape[0]

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden, ot_last = self.decoder.reset_state(batch_size=batch_size)
            hidden = [hidden, ot_last] #2ND version
            
            # Accumulate all batch predictions
            batch_predictions = []
            
            with tf.GradientTape() as tape:
                enc_output = self.encoder(batch_input)

                for t in range(0, seq_len):
                    # using teacher forcing
                    dec_input = tf.expand_dims(batch_target[:, t], 1)
                    
                    predictions, hidden, _, ot_last = self.decoder(dec_input, enc_output, 
                                                              hidden, ot_last)
                    batch_predictions.append(tf.argmax(tf.squeeze(predictions), 
                                                        axis=-1)
                    )
                
                    loss += self.loss_function(batch_target[:, t+1], predictions)

                #batch_loss = (loss / int(targ.shape[0]))
                batch_loss = (loss / seq_len)
                
            batch_acc = acc_metric(batch_target[:, 1:], tf.transpose(batch_predictions),
                    self.pad_id)

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss, batch_acc

        @tf.function
        def _test_step(self, batch_input, batch_target):
            loss = 0
            seq_len = batch_target.shape[1] - 1
            batch_size = batch_target.shape[0]

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden, ot_last = self.decoder.reset_state(batch_size=batch_size)
            hidden = [hidden, ot_last] #2ND version

            # Accumulate all batch predictions
            batch_predictions = []

            enc_output = self.encoder(batch_input)

            for t in range(0, seq_len):
                # using teacher forcing
                dec_input = tf.expand_dims(batch_target[:, t], 1)

                predictions, hidden, _, ot_last = self.decoder(dec_input, enc_output, 
                        hidden, ot_last)
                batch_predictions.append(tf.argmax(tf.squeeze(predictions), 
                    axis=-1)
                )

                loss += self.loss_function(batch_target[:, t+1], predictions)

            batch_loss = (loss / seq_len)
            batch_acc = acc_metric(batch_target[:, 1:], tf.transpose(batch_predictions),
                    self.pad_id)

            return batch_loss, batch_acc

        def train(self, dtrain, dvalid, epochs=10):
            epoch_accs = []
            epoch_losses = []

            batch_size = dtrain.take(1)

            for epoch in range(epochs):
                start = time.time()
                batch_losses = []
                batch_accs = []

                for (batch, (inp, targ)) in enumerate(dtrain):
                    batch_loss, batch_acc = self._train_step(inp, targ)
                    batch_losses.append(batch_loss)
                    batch_accs.append(batch_acc)

                    if batch % 100 == 0:
                        print('Epoch {} Batch {} Loss {:.4f} Acc {:.4f}'.format(epoch + 1,
                                                                                batch,
                                                                                batch_loss.numpy(),
                                                                                batch_acc.numpy()))
                        with self.train_summary_writer.as_default():
                            step = (epoch + 1) * (batch + 1)
                            tf.summary.scalar('loss', batch_loss, step=step)
                            tf.summary.scalar('accuracy', batch_acc, step=step)

                # saving (checkpoint) the model every 2 epochs
                if (epoch + 1) % 2 == 0:
                    print('Saving checkpoint')
                    self.ckpt_manager.save()    
                

                print('Epoch {} Loss {:.4f} - Acc {:.4f}'.format(epoch + 1, 
                                                    np.mean(batch_losses),
                                                    np.mean(batch_accs)))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
                
                
                print('Testing metrics: ')
                test_losses = []
                test_accs = []
                for (batch, (inp, targ)) in enumerate(dvalid):
                    #if inp.shape[0] != BATCH_SIZE: 
                    #    break
                    batch_loss, batch_acc = self._test_step(inp, targ)
                    test_losses.append(batch_loss.numpy())
                    test_accs.append(batch_acc.numpy())

                avg_loss = np.mean(test_losses)
                avg_acc = np.mean(np.mean(test_accs))
                print(f'\t -Loss test: {avg_loss}')
                print(f'\t -Acc test: {avg_acc}')
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', avg_loss, step=(epoch + 1))
                    tf.summary.scalar('accuracy', avg_acc, step=(epoch + 1))
                
                print('\n')
                print('-'*30)
