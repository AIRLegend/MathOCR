import requests
import argparse
import os
import sys
import shutil
import functools
import tarfile
import numpy as np
import pandas as pd

from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.latexlang import LatexTokenizer, Vocabulary

from PIL import Image



def untar_imgs(tarfile_path, extract_path):
    tar = tarfile.open(tarfile_path)
    tar.extractall(extract_path)

def process_images(raw_path, proc_path):
    def check_img(img_path):
        img = np.array(Image.open(proc_path+'formula_images/'+img_path))
        return np.sum(img[img < 200]) > 0

    train_file = raw_path + 'im2latex_train.lst'
    validate_file = raw_path + 'im2latex_validate.lst'
    test_file = raw_path + 'im2latex_test.lst'

    train_df = pd.read_csv(train_file, sep=" ", 
            header=None, names=['ID', 'Image', 'mode'])
    test_df = pd.read_csv(test_file, sep=" ", 
            header=None, names=['ID', 'Image', 'mode'])
    val_df = pd.read_csv(validate_file, sep=" ", 
            header=None, names=['ID', 'Image', 'mode'])

    train_df.Image += '.png'
    test_df.Image += '.png'
    val_df.Image += '.png'

    train_df['is_correct'] = train_df.apply(lambda r: check_img(r[1]), axis=1)
    test_df['is_correct'] = test_df.apply(lambda r: check_img(r[1]), axis=1)
    val_df['is_correct'] = val_df.apply(lambda r: check_img(r[1]), axis=1)
    
    train_df = train_df.loc[train_df['is_correct'] == True]
    test_df = test_df.loc[test_df['is_correct'] == True]
    val_df = val_df.loc[val_df['is_correct'] == True]

    train_file = proc_path + 'im2latex_train.csv'
    validate_file = proc_path + 'im2latex_validate.csv'
    test_file = proc_path + 'im2latex_test.csv'
    
    train_df[['ID', 'Image', 'mode']].to_csv(train_file, index=False)
    test_df[['ID', 'Image', 'mode']].to_csv(test_file, index=False)
    val_df[['ID', 'Image', 'mode']].to_csv(validate_file, index=False)

def build_vocab(formulas, savepath=None):
    base_vocab = """START END PAD SPACE UNK 1 2 3 4 5 6 7 8 9 0 a b c d e f """+\
    """g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O """+\
    """P Q R S T U V W X Y Z { } \\ \\\\ .  , ; _ ^ !  + - * / = !  < > () [ ] ·"""
    base_vocab = base_vocab.split()
    treshold_vocabulary = 40

    tokenizer = LatexTokenizer()
    tokenized_formulas = [tokenizer.tokenize(formula) for formula in formulas]
    corpus = functools.reduce(lambda x,y: x.extend(y) or x, tokenized_formulas)
    counter = Counter(corpus)
    occurrences = {k: v for k,v in sorted(counter.items(), key=lambda item: item[1])} 
    occurrences = {k: v for k, v in occurrences.items() if v >= treshold_vocabulary}
    del(occurrences[' '])  # We already use a special token for the space
    # We want the base vocab to be first in the list and then the 'occurrences'
    char_list = list(occurrences.keys())
    char_list = [t for t in char_list if t not in set(base_vocab)]
    final_vocab = base_vocab + char_list
    
    if savepath:
        with open(savepath, 'w') as file:
            file.write('\n'.join(final_vocab))
    return final_vocab

def process_formulas(rawdir, procdir):
    """ Simply copy the files to the new dir (for the moment) and 
    returns a list with all the formulas merged.
    """
    shutil.copy(rawdir+'im2latex_formulas.lst', 
                procdir+'im2latex_formulas.lst')

    merged_formulas = []
    with open(procdir+'im2latex_formulas.lst', 
              encoding="ISO-8859-1", newline="\n") as file:
        merged_formulas = file.readlines()

    merged_formulas = [e.replace("\n", "") for e in merged_formulas]

    return merged_formulas

def main():
    parser = argparse.ArgumentParser(description='Download im2latex dataset')

    parser.add_argument('--rawdir',
                        type=str,
                        default="../../data/raw/",
                        help="Path where the original dataset is stored.")

    parser.add_argument('--procdir',
                        type=str,
                        default="../../data/processed/",
                        help="Path where the processed dataset will be stored.")


    args = parser.parse_args()

    if not os.path.exists(args.rawdir):
        raise ValueError("The raw data directory does not exist")

    if not os.path.exists(args.procdir):
        raise ValueError("The save directory does not exist")

    print('Cleaning starting. This might take a while...')
    print("Uncompressing formula images...")
    untar_imgs(args.rawdir + 'formula_images.tar.gz', args.procdir)
    print("Processing formulas.")
    formulas = process_formulas(args.rawdir, args.procdir)
    print("Dealing with wrong images.")
    process_images(args.rawdir, args.procdir) 
    print("Building vocabulary file.")
    build_vocab(formulas, savepath=args.procdir+'vocabulary.txt')
    print("Done!")
    
    return 0


if __name__ == "__main__":
    main()
