import re
import time
import argparse
from tqdm import tqdm
from train import Model
from data_utils import FeatureGenerator, Sentence, log_stats
from data_parser import DataParser

"""_summary_
Given a trained model file (and possibly vocabulary file) reads in CoNLL data and writes
CoNLL data where fields 7 and 8 contain dependency tree info.

parse.py should take in a sentence and from the parser configuration
generate labels, which will determine each subsequent parser configuration (from which features can
be determined). Both files, then, should point to a third library file which contains code that, given
configuration at time t ct and label l, determines ct+1.    
"""

LABEL_PATTERN = r'([a-z_]+)(\(([a-z:]+)\))*'
def decompose_pred(pred_label):
    pred_label = re.search(LABEL_PATTERN, pred_label)
    trans_type, _, dep = pred_label.groups()
    return trans_type, dep

def infer_sentence_tree(
        model: Model, s: Sentence, trange: tqdm, verbose,
        drop_blocking_elements
    ):
    num_infers = 0
    while True:
        # getting the features of a current parse state
        s_feats = FeatureGenerator.extract_features(s)
        s_feats = s_feats.reshape((1, len(s_feats)))
        pred_label = model.classify(s_feats)
        trans_type, dep = decompose_pred(pred_label)
        # performing an action
        updated = s.update_state(curr_trans=trans_type, predicted_dep=dep)            

        if verbose:
            trange.set_postfix({
                'trans_count': f'{num_infers}/{2*len(s) + 1} [(2*tokens) + 1]',
                'prev_trans': pred_label,
                'stack': f'{[t.word for t in s.stack]}',
            })
            num_infers += 1
            
        # Tweak for better chance on catching correct dependancies:
        # Drop blocking elements and continue classification
        if not updated:
            if drop_blocking_elements:
                if len(s.stack) > 1:
                    dropped_token = s.stack.pop(-1)
                    # assign arbitirarly something (head is previous token)
                    dropped_token.head = str(int(dropped_token.token_id) - 1)
                elif len(s.buffer) > 0:
                    s.stack.append(s.buffer.pop(0))    
            else:
                break
        
        if s.is_exausted():
            break
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser file configuration.')
    parser.add_argument('-m', default='train.model', type=str, help='model file including vocab (encoding)')
    parser.add_argument('-i', default='parse.in', type=str, help='input filepath')
    parser.add_argument('-o', default='parse.out', type=str, help='output filepath')
    parser.add_argument('-trans', default='std', type=str, help='transition system')
    parser.add_argument('-verbose', default=False, type=bool, help='verbose')
    parser.add_argument('-dropb', default=True, type=bool, help='whether to drop blocking elements while transiting')
    args = parser.parse_args()
    
    if args.dropb:
        print('Dropping blocking elements')
    
    sentences = DataParser.read_parse_tree(args.i, transition_system=args.trans)
    log_stats(sentences)
    
    # Ensure unlabeled
    def unlabel_sentence(s):
        for t in s.tokens:
            t.head = 0
            t.dep = 0
        return s
    
    # Ensure empty head and dep
    sentences = list(map(lambda x: unlabel_sentence(x), sentences))

    model = Model.load_model(args.m)
    
    sentences_trange = tqdm(sentences, desc='Trees/Sentences')
    if not args.verbose:
        print('\nStopped verbosing!')
        sentences_trange.close()
    
    infer_stime = time.time()
    for s in sentences_trange:
        infer_sentence_tree(model, s, sentences_trange,
                            args.verbose,
                            drop_blocking_elements=args.dropb)
    print(f'Finshed infering sentences trees in {time.time() - infer_stime: < .2f}s')
   
    # write CoNLL formatted file with depend tree info aka. field 7 & 8
    DataParser.update_conll_file(sentences, args.i, args.o)
    print(f'Finished writing updated CoNLL file as {args.o}')
