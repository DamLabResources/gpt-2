#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import fire
import json
import os
import numpy as np
import tensorflow as tf
import random
import time
#import tqdm
import glob

import encoder
import docx


def extract_text(doc_path):

    doc = docx.Document(doc_path)

    raw_text = ''
    for par in doc.paragraphs:
        raw_text += par.text

    return raw_text


def load_dataset(enc, path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    found = 0
    total_chars = 0
    for path in paths:

        #if path.endswith('.txt'):
        #    with open(path, 'r') as fp:
        #        raw_text = fp.read()
        if path.endswith('.docx'):
            raw_text = extract_text(path)
        else:
            continue

        if raw_text:
            print(path)
            found += 1
            total_chars += len(raw_text)
            tokens = np.stack(enc.encode(raw_text))
            token_chunks.append(tokens)

            #if found > 200:
            #    print('Dropping after %i chars' % total_chars)
            #    break

    return token_chunks


def encode_main(in_text, out_npz, model_name='117M'):
    enc = encoder.get_encoder(model_name)
    print('Reading files')
    chunks = load_dataset(enc, in_text)
    print('Writing', out_npz)
    np.savez_compressed(out_npz, *chunks)


if __name__ == '__main__':
    fire.Fire(encode_main)
