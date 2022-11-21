import numpy as np
import pandas as pd
import tqdm as tqdm
from data_utils import Sentence


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
                    # breakpoint()
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
