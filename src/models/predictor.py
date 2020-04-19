import tensorflow as tf
import numpy as np

from .encoder import Encoder
from .decoder import Decoder

from utils import image as improc
from utils.latexlang import LatexTokenizer
from utils.search import beam_search


class Im2SeqPredictor:
    """ Wrapper of the Encoder/Decoder models which encapsulates
    the logic to make precitions (either greddy or beam search)
    """

    def __init__(self, encoder_weights, decoder_weights, vocabulary):
        
        self.vocabulary = vocabulary
        self.tokenizer = LatexTokenizer()

        self.encoder = Encoder()
        self.decoder = Decoder(40, 256, len(self.vocabulary))

        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights

        # Build pipeline
        self._build_models()

    def predict(self, image, beam=False):
        image = improc.process(image)

        if beam:
            formula = self._predict_beam(image)
        else:
            formula = self._predict_greedy(image)

        formula = self.tokenizer.decode(formula, self.vocabulary)

        return formula
    
    def _build_models(self):
        encoded = self.encoder(tf.random.uniform((1, 75, 300, 1)))
        hidden, ot_last = self.decoder.reset_state(1)
        hidden = [hidden, ot_last]
        last_charac = tf.expand_dims([0], axis=-1)
        self.decoder(last_charac, encoded, hidden, ot_last)

        self.encoder.load_weights(self.encoder_weights)
        self.decoder.load_weights(self.decoder_weights)

    def _predict_beam(self, image, depth=3, beam_width=3):
        formula = [self.vocabulary['START']]
        encoded_image = image#_process_image(image)
        encoded_image = self.encoder(tf.expand_dims(encoded_image, axis=0))
        last_charac = formula[-1]
        state = None

        newstate, o_t_last = self.decoder.reset_state(1)
        newstate = [newstate, o_t_last]

        for i in range(30): #TODO: Replacewith while
            #print(i)
            levels = []
            indices = formula[-1]

            for j in range(depth):

                predictions, newstate, _, o_t_last = self.decoder (
                    tf.reshape(indices, (beam_width**(j),1)),
                    tf.repeat(encoded_image, beam_width**(j), axis=0),
                    newstate,
                    o_t_last
                )

                newstate[0] = tf.repeat(newstate[0], beam_width, axis=0)
                newstate[1] = tf.repeat(newstate[1], beam_width, axis=0)
                o_t_last = tf.repeat(o_t_last, beam_width, axis=0)

                # Expand top-n
                probs, indices = tf.math.top_k(predictions, k=beam_width)
                indices = tf.reshape(indices, (-1,))
                probs = tf.reshape(probs, (-1,))

                levels.append(list(zip(indices.numpy().tolist(), probs.numpy().tolist())))

                best_path, last_idx = beam_search(levels, index=True)
                best_path = [i[0] for i in best_path]

                if j >= depth -1:
                    # Save the last states for continuing with the predictions.
                    # depth ^ beah_width states for the last posibilities of the tree.
                    # Once we have the best path, we return the state of that path.
                    #last_state = newstate[last_idx]
                    newstate = (
                        tf.expand_dims(newstate[0][last_idx], axis=0),
                        tf.expand_dims(newstate[1][last_idx], axis=0)
                    )
                    o_t_last = tf.expand_dims(o_t_last[last_idx], axis=0)

            formula.extend(best_path)

            if self.vocabulary['END'] in best_path:
                # Select up to the first occurrence
                formula = formula[:formula.index(self.vocabulary['END']) + 1]
                break

        return formula

    def _predict_greedy(self, image):

        if len(image.shape) < 4:
            # User passes a single image, not a batch
            image = tf.expand_dims(image, axis=0)

        encoded = self.encoder(image)

        batch_size = 1

        formula = [self.vocabulary['START']]

        # Reset state
        hidden, ot_last = self.decoder.reset_state(batch_size)
        hidden = [hidden, ot_last] 

        for i in range(90):
            #charac_inp = np.expand_dims([last_charac], axis=0)
            charac_inp = np.expand_dims([formula[-1]], axis=0)
            predictions, hidden, _, ot_last = self.decoder(charac_inp,
                                                           encoded,
                                                           hidden,
                                                           ot_last)
            predictions = tf.squeeze(predictions)
            newchar = int(tf.argmax(predictions, axis=0).numpy())
            formula.append(newchar)

            if newchar is self.vocabulary['END']:
                break

        return formula

