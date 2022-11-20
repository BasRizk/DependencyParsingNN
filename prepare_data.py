import argparse
from data_utils import FeatureGenerator, log_stats
from data_parser import DataParser

"""_summary_
preparedata.py converts CoNLL data (train and dev) into features of the parser configuration paired
with parser decisions. 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparing Data Configuration.')
    parser.add_argument(
        '-f','--files', action='store', dest='data_files', type=str,
        nargs='+', 
        default=["train.orig.conll", "dev.orig.conll"],
    )    
    parser.add_argument('-trans', default='std', type=str, help='transition system')
    args = parser.parse_args()
    
    if args.trans == 'std':
        print('Using arc-standard transition system')
    else:
        print('Using arc-eager transition system')
    
    features_generator = FeatureGenerator()
    for filepath in args.data_files:
        sentences = DataParser.read_parse_tree(filepath, transition_system=args.trans)[1:]
        log_stats(sentences)
        
        print(f'Finished reading {filepath} file')
        train_dataset =\
            features_generator.generate_labeled_dataset(
                sentences
                )
        print("Finished generating features") 
        out_filepath = f'{filepath.split(".")[0]}.converted'
        train_dataset.to_csv(out_filepath, index=False)
        print(f"Written {out_filepath}")        
      