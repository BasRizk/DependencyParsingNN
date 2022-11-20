import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'

class Token:
    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.head = head
        self.dep = dep
        self.lc, self.rc = [], []
        
    def get_left_most_child(self, num=1):
        return self.lc[0 + num - 1] if len(self.lc) >= num else NULL_TOKEN
      
    def get_right_most_child(self, num=1):
        return self.rc[-1 - num + 1] if len(self.rc) >= num else NULL_TOKEN

    def point_to_unk(self):
        self.head = UNK_TOKEN.token_id
        self.dep = UNK
            
    def __str__(self):
        return f"{self.token_id:5} | {self.word} | {self.pos} |" +\
            f" {self.head} | {self.dep}"

ROOT_TOKEN = Token(token_id="0", word=ROOT, pos=ROOT, head="-1", dep=ROOT)
NULL_TOKEN = Token(token_id="-1", word=NULL, pos=NULL, head="-1", dep=NULL)
UNK_TOKEN = Token(token_id="-1", word=UNK, pos=UNK, head="-1", dep=UNK)


class Sentence:

    def __init__(self, tokens=[]):
        self.root = Token(token_id="0", word=ROOT, pos=ROOT, head="-1", dep=ROOT)
        self.tokens = tokens.copy()
        self.stack = [ROOT_TOKEN]
        self.buffer = tokens.copy()
        self.arcs = []
    
    def __len__(self):
        return len(self.tokens)
    
    def is_exausted(self):
        return len(self.stack) == 1 and len(self.buffer) == 0    
    
    def add_token(self, token):
        self.tokens.append(token)
        self.buffer.append(token)
        
    def peek_stack(self, top=1):
        items = list(reversed(self.stack[-top:]))
        if len(self.stack) < top:
            items += [NULL_TOKEN for _ in range(top - len(self.stack))]
        return items
    
    def peek_buffer(self, top=1):
        items =  self.buffer[:top]
        if len(self.buffer) < top:
            items += [NULL_TOKEN for _ in range(top - len(self.buffer))]
        return items

    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation from [shift, left_arc, or right_arc] """
        for operation in ['left_arc', 'right_arc', 'shift']:
            # Retrive transition name completely
            trans = self._check_trans(operation)
            if trans is not None:
                return trans
        return None

    def _is_dep_in_buff(self, token_id):
        for t in self.buffer:
            if t.head == token_id:
                return True
        return False

    def _check_trans(self, potential_trans):
        """ checks if transition can legally be performed"""
        # LEFT, top of stack is parent of second-top
        def check_left_arc_sat():
            if self.stack[-1].token_id != self.stack[-2].head:
                return None
            return f"left_arc({self.stack[-2].dep})"

        # RIGHT, second-top of stack is parent of top, 
        # and no depends of top in buffer (buff is empty)        
        def check_right_arc_sat():
            if self._is_dep_in_buff(self.stack[-1].token_id) or\
                self.stack[-2].token_id != self.stack[-1].head:
                return None
            return f"right_arc({self.stack[-1].dep})"

        if len(self.stack) >= 2:
            if potential_trans == 'left_arc':
                return check_left_arc_sat()
            if potential_trans == 'right_arc':
                return check_right_arc_sat()
        if potential_trans == 'shift' and len(self.buffer) >= 1:
            return 'shift'
        return None
    
    def update_state(self, curr_trans, predicted_dep=None):
        """ 
        updates the sentence according to the given 
        transition assuming dependancy satisfiability
        but NOT legality
        """
        
        if 'shift' in curr_trans:
            if len(self.buffer) == 0:
                return False
            self.stack.append(self.buffer.pop(0))
            return True
        
        if len(self.stack) < 2:
            return False
        
        if 'left_arc' in curr_trans:
            parent = self.stack[-1]
            child = self.stack.pop(-2)
            parent.lc.insert(0, child)
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'l'))
            return True
            
        if 'right_arc' in curr_trans:
            parent = self.stack[-2]
            child = self.stack.pop(-1)
            parent.rc.append(child)
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'r'))
            return True
            
    def __str__(self) -> str:
        return f"Stack {[t.word for t in self.stack]} | " +\
            f"Buffer {[t.word for t in self.buffer]}"

class FeatureGenerator:
    
    def __init__(self):
        selections = [
            's_1', 's_2', 's_3',
            'b_1', 'b_2', 'b_3',
            'lc_1(s_1)', 'rc_1(s_1)', 'lc_2(s_1)', 'rc_2(s_1)',
            'lc_1(s_2)', 'rc_1(s_2)', 'lc_2(s_2)', 'rc_2(s_2)',
            'lc_1(lc_1(s_1))', 'rc_1(rc_1(s_1))',
            'lc_1(lc_1(s_2))', 'rc_1(rc_1(s_2))'
        ]
        self.columns = [f'{name}' for name in selections]
        self.columns += [f'pos {name}' for name in selections]
        self.columns += [f'dep {name}' for name in selections[6:]]
            
    def generate_labeled_dataset(self, sentences):
        datapoints = []
        num_dropped_sentences = 0
        for sentence in tqdm(sentences, desc='Sentences'):
            possible_configs = self.generate_possible_configs(sentence)
            if len(possible_configs) > 0:
                datapoints += possible_configs
            else:
                num_dropped_sentences += 1
        print(f'Dropped configs of {num_dropped_sentences} sentences of {len(sentences)}.')
        datapoints =\
            pd.DataFrame(datapoints, columns = self.columns + ['label'])
        return datapoints
    
    def generate_possible_configs(self, sentence: Sentence):
        configs = []
        while True:
            features = self.extract_features(sentence)
            trans = sentence.get_trans()
            if trans is None:
                if len(sentence.stack) > 1:
                    # is not projective
                    configs = []
                break
            configs.append(np.append(features, [trans]))
            sentence.update_state(trans)        
        return configs            
    
    @classmethod
    def extract_features(cls, sentence):
        """ returns the features for a sentence parse configuration """
        # Use embedding generated in extract_features
        word_feat = cls.get_features_attr(
            sentence, lambda t : t.word
        )
        pos_feat = cls.get_features_attr(
            sentence, lambda t : t.pos
        )
        dep_feat = cls.get_features_attr(
            sentence, lambda t : t.dep, get_parents=False
        )
    
        return np.concatenate([word_feat, pos_feat, dep_feat])
    
    @classmethod
    def get_features_attr(cls, sentence, getter_func, get_parents=True):
        features = []
        if get_parents:
            # s_1, s_2, s_3
            features += [getter_func(t) for t in sentence.peek_stack(3)]
            # b_1, b_2, b_3
            features += [getter_func(t) for t in sentence.peek_buffer(3)]
            
        # lc_1(s_i), rc_1(s_i), lc_2(s_i), rc_2(s_i) for i = 1, 2
        for t in sentence.peek_stack(2):
            features += [
                getter_func(t.get_left_most_child()),
                getter_func(t.get_right_most_child()),
                getter_func(t.get_left_most_child(2)),
                getter_func(t.get_right_most_child(2))
            ]
            
        # lc_1(lc_1(s_i)), rc_1(rc_1(s_i)) for i = 1, 2
        for t in sentence.peek_stack(2):
            features += [
                getter_func(t.get_left_most_child().get_left_most_child()),
                getter_func(t.get_right_most_child().get_right_most_child())
            ]
            
        return features
    
    
class BasicWordEncoder:
    def __init__(self, training_df) -> None:
        unique_words = pd.unique(training_df[training_df.columns[:-1]].values.ravel('K'))
        self.word_to_idx = {w: i+1 for i, w in enumerate(unique_words)}
        self.word_to_idx[UNK] = 0
        self.unique_labels = pd.unique(training_df[training_df.columns[-1]])
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.label_to_idx[UNK] = 0

    def _encode_word(self, word):
        idx = self.word_to_idx.get(word)
        if idx is None:
            idx = self.word_to_idx[UNK]
        return idx
    
    def _encode_label(self, label):
        idx = self.label_to_idx.get(label)
        if idx is None:
            idx = self.label_to_idx[UNK]
        return idx
        
    def decode_label(self, label_embed):
        return self.unique_labels[label_embed]
    
    def encode_dataset(self, df, labeled=True):
        if labeled:
            df[df.columns[:-1]] = df[df.columns[:-1]].applymap(lambda x: self._encode_word(x))
            df[df.columns[-1:]] = df[df.columns[-1:]].applymap(lambda x: self._encode_label(x))
        else:
            df = df.applymap(lambda x: self._encode_word(x))
            df = df.applymap(lambda x: self._encode_label(x))
        return df
    
    def encode_features_vector(self, feats):
        return np.vectorize(self._encode_word)(feats)

    def get_dictionary_size(self):
        return len(self.word_to_idx)
    
    def get_num_of_labels(self):
        return len(self.unique_labels)


def log_stats(sentences):
    num_tokens_per_s = list(map(lambda x: len(x.tokens), sentences))
    print('Token stats:')
    print(f'# of sentences: {len(sentences)}')
    print(f'mean: {np.mean(num_tokens_per_s)}, median: {np.median(num_tokens_per_s)} std: {np.std(num_tokens_per_s)}')

