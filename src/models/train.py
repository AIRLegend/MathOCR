import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from models.encoder import Encoder
from models.decoder import Decoder
from models.trainer import Im2SeqTrainer
from utils.latexlang import Vocabulary
from data.dataset import build_im2latex


def main():

    parser = argparse.ArgumentParser(description='Train Image to Sequence based model.')

    parser.add_argument('--processed_data',
                        type=str,
                        default="../../data/processed/",
                        help="Path to processed data.")

    parser.add_argument('--savedir',
                        type=str,
                        default="../../models/",
                        help="Diretory to save model weights and training logs.")

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="Number of epochs to train the model")

    parser.add_argument('--bs',
                        type=int,
                        default=16,
                        help="Batch size")

    parser.add_argument('--restore',
                        help="""Restores last checkpoint saved if it exists.""",
                        action='store_true')

    args = parser.parse_args()

    restore_checkpoint = False
    if args.restore is True:
        restore_checkpoint = True

    processed_data = args.processed_data
    savedir = args.savedir

    if not processed_data.endswith("/"):
        processed_data += '/'

    if not savedir.endswith("/"):
        savedir += '/'

    vocabulary_path = processed_data + 'vocabulary.txt'
    images_path = processed_data + 'formula_images/'
    formulas_path = processed_data + 'formulas.lst'
    csvs_path = processed_data
    checkpoint_savedir = savedir + 'checkpoints/'
    logdir = savedir + 'logs/'
    weights_path = savedir

    vocab = Vocabulary.load_from(vocabulary_path)

    dtrain, dtest, dvalid = build_im2latex(formulas_path,
                                           images_path,
                                           csvs_path,
                                           vocab,
                                           batch_size=args.bs)

    encoder = Encoder()

    decoder = Decoder(40, 256, len(vocab))

    trainer = Im2SeqTrainer(encoder,
                            decoder,
                            vocab['PAD'],
                            logdir=logdir,
                            savedir=checkpoint_savedir,
                            restore_checkpoint=restore_checkpoint,
                            optimizer=None)

    trainer.train(dtrain, dvalid, epochs=args.epochs)

    encoder.save_weights(weights_path + 'encoder.h5')
    decoder.save_weights(weights_path + 'decoder.h5')


if __name__ == "__main__":
    main()
