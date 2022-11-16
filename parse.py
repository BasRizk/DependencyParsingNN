import argparse
from data_utils import BasicWordEncoder
from data_utils import Sentence, Token
from train import Model

"""_summary_
Given a trained model file (and possibly vocabulary file) reads in CoNLL data and writes
CoNLL data where fields 7 and 8 contain dependency tree info.

TODO High-level functions you may want to have in your parser class include initializing to the
start state for a sentence, performing an action, getting the features of a current parse state, getting
the ground truth action. The provided skeleton is one such way to develop your code, though more
supporting functions are almost certainly a good idea.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser file configuration.')
    parser.add_argument('-m', type=str, help='model file including vocab (encoding)')
    parser.add_argument('-i', type=str, help='input file')
    parser.add_argument('-o', type=str, help='output file')
    args = parser.parse_args()
    
    model = Model.load_model(args.m)
    
    # TODO Read data from input file args.i
    # TODO Preprocess input data
    # parse.py should take in a sentence and from the parser configuration
    # generate labels, which will determine each subsequent parser configuration (from which features can
    # be determined). Both files, then, should point to a third library file which contains code that, given
    # configuration at time t ct and label l, determines ct+1.
    # TODO use model properly    
    # TODO output CoNLL formatted file with depend tree info aka. field 7 & 8
    
