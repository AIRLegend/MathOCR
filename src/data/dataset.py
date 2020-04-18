import pandas as pd
import numpy as np
import tensorflow as tf

from utils.latexlang import LatexTokenizer
from utils.image import process as im_proc

def _parse_dataset(filename, target):
    image = tf.image.decode_png(tf.io.read_file(filename), channels=1)
    image = im_proc(image)

    return image, target

def build_im2latex(path_formulas, path_images, path_csvs, vocabulary, batch_size=16):

    if not path_images.endswith('/'):
        path_images += '/'


    tokenizer = LatexTokenizer()

    formulas = []
    with open(path_formulas, encoding="ISO-8859-1", newline="\n") as file:
        formulas = file.readlines()

    formulas = [e.replace("\n", "") for e in formulas]
    tokenized_formulas = [tokenizer.tokenize(formula) for formula in formulas]
    encoded_formulas = [tokenizer.encode(t, vocabulary) for t in tokenized_formulas]

    #Use 90 as max length
    padded_formulas = tf.keras.preprocessing.sequence.\
            pad_sequences(encoded_formulas, 90, padding='post', 
                          truncating='post',
                          value=vocabulary['PAD'])

    train_df = pd.read_csv(path_csvs + 'im2latex_train.csv')
    test_df = pd.read_csv(path_csvs + 'im2latex_test.csv')
    val_df = pd.read_csv(path_csvs + 'im2latex_validate.csv')

    train_df.Image = path_images + train_df.Image
    test_df.Image = path_images + test_df.Image
    val_df.Image = path_images + val_df.Image


    # Split formulas into train/test/val
    train_formulas = []
    for r in train_df.iterrows():
        train_formulas.append(padded_formulas[r[1].ID])

    test_formulas = []
    for r in test_df.iterrows():
        test_formulas.append(padded_formulas[r[1].ID])
        
    val_formulas = []
    for r in val_df.iterrows():
        val_formulas.append(padded_formulas[r[1].ID])

    dtrain = tf.data.Dataset.from_tensor_slices((train_df.Image, train_formulas))
    dtrain = dtrain.map(_parse_dataset)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .batch(batch_size, drop_remainder=True)

    dtest = tf.data.Dataset.from_tensor_slices((test_df.Image, test_formulas))
    dtest = dtest.map(_parse_dataset)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .batch(batch_size, drop_remainder=True)

    dvalid = tf.data.Dataset.from_tensor_slices((val_df.Image, val_formulas))
    dvalid = dvalid.map(_parse_dataset)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .batch(batch_size, drop_remainder=True)

    return dtrain, dtest, dvalid
