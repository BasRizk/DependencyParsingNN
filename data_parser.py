
from typing import List
from data_utils import Token, Sentence
import pandas as pd
import numpy as np
"""_summary_

Returns:

Each line is in CoNLL data format, and has 10 tab-separated fields as follows:
1. current sentence word number (1-based)
2. the word
3. lemma form of the word
4. coarse-grained POS tag
5. fine-grained POS tag (in practice, both of these are the same)
6. morphological features (in practice, just the token ‘-’)
7. token id of this word’s parent (head),
or 0 if this word’s parent is the root. 
If this data has not yet been provided 
(i.e. the sentence isn’t parsed) it is ’-’.
8. dependency label for the relation of this word to its parent.
If this data has not yet been provided (i.e.
the sentence isn’t parsed) it is ’-’.
9. other dependencies (in practice, just ‘-’)
10. other info (in practice, just ‘-’)
################################################################################

"""

class DataParser:
    
    @staticmethod
    def split_conll_line(file):
        line =  file.readline().strip()
        if line:
            return line.split("\t")
        return line
    
    @staticmethod
    def write_conll_line(file, line):
        file.write("\t".join(line) + "\n")
    
    @staticmethod
    def read_parse_tree(filepath="train.orig.conll", transition_system='std'):        
        with open(filepath) as parseTreesFile:
            sentences = []
            while True:
                line = DataParser.split_conll_line(parseTreesFile)
                if not line:
                    # File is over
                    break
                sentence_tokens = []
                while(True):
                    token = Token(
                        token_id = line[0],
                        word = line[1],
                        pos = line[3],
                        head = line[6],
                        dep = line[7],
                    )
                    sentence_tokens.append(token)
                    line = DataParser.split_conll_line(parseTreesFile)
                    if not line:
                        # Sentence is over
                        sentences.append(
                            Sentence(sentence_tokens, transition_system=transition_system)
                        )
                        break
                    
        return sentences
    
    
    @staticmethod
    def update_conll_file(
            sentences: List[Sentence],
            conll_filepath: str,
            output_filepath,
            transition_system='std'
        ):
        
        head_dep_updates = np.concatenate(list(map(
            lambda x: [np.array([t.head, t.dep]) for t in x.tokens], 
            sentences
        )))
        
        head_dep_updates_pt = 0
        
        new_conll_file = open(output_filepath, mode='w')
        with open(conll_filepath) as old_conll_file:
            sentences = []
            while True:
                line = DataParser.split_conll_line(old_conll_file)
                if not line:
                    # File is over
                    new_conll_file.write('\n')
                    break
                while(True):
                    
                    line[6], line[7] = head_dep_updates[head_dep_updates_pt]
                    head_dep_updates_pt += 1
                    DataParser.write_conll_line(new_conll_file, line)
                    line = DataParser.split_conll_line(old_conll_file)
                    if not line:
                        # Sentence is over
                        new_conll_file.write('\n')
                        break
        new_conll_file.close()
    
        # conll_df = pd.read_csv(conll_filepath, header=None, delimiter="\t")

        # assert len(head_dep_updates) == len(conll_df)
        # conll_df[[6,7]] = head_dep_updates
        
        # # Write updated conll_df back into conll form
        # conll_df.to_csv(output_filepath, sep='\t', index=False, header=None)
            
                