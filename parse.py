import re
import argparse
from tqdm import tqdm

from train import Model
from data_utils import FeatureGenerator, Sentence
from data_parser import DataParser

"""_summary_
Given a trained model file (and possibly vocabulary file) reads in CoNLL data and writes
CoNLL data where fields 7 and 8 contain dependency tree info.

parse.py should take in a sentence and from the parser configuration
generate labels, which will determine each subsequent parser configuration (from which features can
be determined). Both files, then, should point to a third library file which contains code that, given
configuration at time t ct and label l, determines ct+1.    
"""

LABEL_PATTERN = r'([a-z_]+)(\(([a-z]+)\))*'
def decompose_pred(pred_label):
    pred_label = re.search(LABEL_PATTERN, pred_label)
    trans_type, _, dep = pred_label.groups()
    return trans_type, dep

def infer_sentence_tree(model: Model, s: Sentence, trange: tqdm):
    num_infers = 0
    while True:
        # getting the features of a current parse state
        s_feats = FeatureGenerator.extract_features(s)
        s_feats = s_feats.reshape((1, len(s_feats)))
        pred_label = model.classify(s_feats)
        trans_type, dep = decompose_pred(pred_label)
        
        # performing an action
        updated = s.update_state(curr_trans=trans_type, predicted_dep=dep)
        
        # Tweak for better chance on catching correct dependancies:
        # Drop blocking elements and continue classification
        if not updated:
            if len(s.stack) > 1:
                s.stack.pop(-1)
            elif len(s.buffer) > 0:
                s.stack.append(s.buffer.pop(0))
                
        trange.set_postfix({
            'trans_count': f'{num_infers}/{2*len(s) + 1} [(2*tokens) + 1]',
            'prev_trans': pred_label,
            'stack': f'{[t.word for t in s.stack]}',
        })
        num_infers += 1
        if s.is_exausted():
            break
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser file configuration.')
    parser.add_argument('-m', default='train.model', type=str, help='model file including vocab (encoding)')
    parser.add_argument('-i', default='test.sample.conll', type=str, help='input filepath')
    parser.add_argument('-o', default='pred.sample.conll', type=str, help='output filepath')
    args = parser.parse_args()

    sentences = DataParser.read_parse_tree(args.i)
    model = Model.load_model(args.m)
    
    sentences_trange = tqdm(sentences, desc='Trees/Sentences')
    for s in sentences_trange:
        infer_sentence_tree(model, s, sentences_trange)
    
    # write CoNLL formatted file with depend tree info aka. field 7 & 8
    DataParser.update_conll_file(sentences, args.i, args.o)
    print(f'Finished writing updated CoNLL file as {args.o}')
