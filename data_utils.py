import numpy as np
import pandas as pd
from tqdm import tqdm

P_PREFIX = '<p>'
L_PREFIX = '<l>'
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'


class Token:
    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = int(token_id)
        self.word = word
        self.pos = pos
        self.head = int(head)
        self.dep = dep
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []
        
    def reset_states(self):
        self.predicted_head = -1
        self.predicted_dep = '<null>'
        self.lc, self.rc = [], []
    
    def get_left_most_child(self, num=1):
        return self.lc[0 + num - 1] if len(self.lc) >= num else NULL_TOKEN
      
    def get_right_most_child(self, num=1):
        return self.rc[-1 - num + 1] if len(self.rc) >= num else NULL_TOKEN
            
    def __str__(self):
        return f"{self.token_id:5} | {self.word} | {self.pos} |" +\
            " {self.head} | {self.dep}"

ROOT_TOKEN = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
NULL_TOKEN = Token(token_id=-1, word=NULL, pos=NULL, head=-1, dep=NULL)
UNK_TOKEN = Token(token_id=-1, word=UNK, pos=UNK, head=-1, dep=UNK)


class Sentence:

    def __init__(self, tokens=[]):
        self.root = Token(token_id=0, word=ROOT, pos=ROOT, head=-1, dep=ROOT)
        self.tokens = tokens.copy()
        self.stack = [ROOT_TOKEN]
        self.buffer = tokens.copy()
        self.arcs = []
        self.predicted_arcs = []
    
    def add_token(self, token):
        self.tokens.append(token)
        self.buffer.append(token)
        
    # def parse_arcs(self):
    #     while True:
    #         curr_trans = self.get_trans()
    #         if not curr_trans:
    #             break
    #         self.update_state(curr_trans=curr_trans)

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
    
    def is_projective(self):
        """ determines if sentence is projective when ground truth given """
        # tree is non-projective if it has crossing arcs. 
        while True:
            trans = self.get_trans()
            if not trans:
                if len(self.stack) > 1:
                    return False
                break
            self.update_state(trans)  
        return True  

    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation from [shift, left_arc, or right_arc] """
        for trans in ['left_arc', 'right_arc', 'shift']:
            if self.check_trans(trans):
                return trans
        return None


    def _is_dep_in_buff(self, token_id):
        for t in self.buffer:
            if t.head == token_id:
                return True
        return False

    def check_trans(self, potential_trans):
        """ checks if transition can legally be performed"""
        # LEFT, top of stack is parent of beneth it
        def check_left_arc():
            if len(self.stack) < 2 or\
                self.stack[-1].token_id != self.stack[-2].head:
                return False
            return True
        
        # RIGHT, beneth top of stack is parent of top, 
        # and no depends of top in buffer (buff is empty)        
        def check_right_arc():
            if self._is_dep_in_buff(self.stack[-1].token_id) or\
                len(self.stack) < 2 or\
                self.stack[-2].token_id != self.stack[-1].head:
                return False
            return True

        if potential_trans == 'left_arc' and check_left_arc() or\
           potential_trans == 'right_arc' and check_right_arc() or\
           potential_trans == 'shift' and len(self.buffer) >= 1:
            return True
        return False
    
    def update_state(self, curr_trans, predicted_dep=None):
        """ updates the sentence according to the given transition (may or may not assume legality, you implement) """
        if curr_trans == 'shift':
            self.stack.append(self.buffer.pop(0))
            
        elif curr_trans == 'left_arc':
            parent = self.stack[-1]
            child = self.stack.pop(-2)
            parent.lc.insert(0, child)
            self.arcs.append((parent, child, child.dep, 'l'))
            
        elif curr_trans == 'right_arc':
            parent = self.stack[-2]
            child = self.stack.pop(-1)
            parent.rc.append(child)
            self.arcs.append((parent, child, child.dep, 'r'))
            
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

    #     self.vocab_size = vocab_size
    #     self.word_to_idx = {}
    #     self.word_to_idx[ROOT] = 0    
    #     self.word_to_idx[UNK] = 1
    #     self.idx_token_counter = 2    
        
    # def encode(self, word, token=False):
    #     if token and len(self.word_to_idx) > self.idx_token_counter:
    #         return self.word_to_idx[UNK]
        
    #     idx = self.word_to_idx.get(word)
    #     if idx is None:
    #         idx = self.word_to_idx[word] = self.idx_token_counter
    #         self.idx_token_counter += 1
    #     return idx
            
    def generate_dataset(self, sentences):
        datapoints = []
        for sentence in tqdm(sentences, desc='Sentences'):
            datapoints += self.generate_possible_configs(sentence)
        datapoints =\
            pd.DataFrame(datapoints, columns = self.columns + ['label'])
        return datapoints
    
    def generate_possible_configs(self, sentence: Sentence):
        configs = []
        while True:
            word_feat, pos_feat, dep_feat = self.extract_features(sentence)
            trans = sentence.get_trans()
            if not trans:
                if len(sentence.stack) > 1:
                    # is projective
                    configs = []
                break
            configs += [word_feat + pos_feat + dep_feat + [trans]]
            sentence.update_state(trans)        
        return configs            
        

        
    def extract_features(self, sentence):
        """ returns the features for a sentence parse configuration """
        # Use embedding generated in extract_features
        word_features = self.get_features_attr(
            sentence, lambda t : t.word
        )
        pos_features = self.get_features_attr(
            sentence, lambda t : t.pos
        )
        dep_features = self.get_features_attr(
            sentence, lambda t : t.dep, get_parents=False
        )
        

        return word_features, pos_features, dep_features
        
    def get_features_attr(self, sentence, getter_func, get_parents=True):
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