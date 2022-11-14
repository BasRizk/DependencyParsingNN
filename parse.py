# TODO • parse.py, given a trained model file (and possibly vocabulary file) reads in CoNLL data and writes
# CoNLL data where fields 7 and 8 contain dependency tree info. The syntax of the file should be python
# parse.py -m [modelfile] -i [inputfile] -o [outputfile].



# High-level functions you may want to have in your parser class include initializing to the
# start state for a sentence, performing an action, getting the features of a current parse state, getting
# the ground truth action. The provided skeleton is one such way to develop your code, though more
# supporting functions are almost certainly a good idea.

# parse.py should take in a sentence and from the parser configuration
# generate labels, which will determine each subsequent parser configuration (from which features can
# be determined). Both files, then, should point to a third library file which contains code that, given
# configuration at time t ct and label l, determines ct+1.

