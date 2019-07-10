#!/usr/bin/env python
# coding: utf8
"""Load vectors for a language trained using fastText
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals
import plac
import numpy

import spacy

@plac.annotations(
    model_dir=('Model output directory', 'option', 'm'))
def main(lang: 'Spacy language', vectors_loc: 'Vectors location', model_dir='model'):
    """
    ex)
        python load_fastText.py en vector/wiki.en.vec
        python load_fastText.py en vector/wiki.en.vec -m ./model
    """

    # create empty language class – this is required if you're planning to
    # save the model to disk and load it back later (models always need a
    # "lang" setting). Use 'xx' for blank multi-language class.
    nlp = spacy.blank(lang)
    with open(vectors_loc, 'rb') as file_:
        header = file_.readline()
        nr_row, nr_dim = header.split()
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in file_:
            line = line.rstrip().decode('utf8')
            pieces = line.rsplit(' ', int(nr_dim))
            word = pieces[0]
            vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
            nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab

    nlp.to_disk(model_dir)

if __name__ == '__main__':
    plac.call(main)