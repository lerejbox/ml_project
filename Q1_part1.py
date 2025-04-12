from Q1_functions import estimate_emission_parameters, estimate_emission_parameters_with_smoothing
import argparse

def read_train_file(file_path):
    """Parse the training file into a list of (word, tag) pairs."""
    train_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try tab separator first
            parts = line.split('\t')
            
            # Fall back to space separator if tab doesn't yield 2 parts
            if len(parts) != 2:
                parts = line.split(' ', 1)
                
            if len(parts) == 2:
                word, tag = parts
                train_data.append((word, tag))
    
    return train_data


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Estimate emission parameters for POS tagging')
    parser.add_argument('--smoothing', action='store_true', help='Use smoothing for rare words')
    parser.add_argument('--k', type=int, default=3, help='Threshold for rare words in smoothing (default: 3)')
    parser.add_argument('--train', type=str, default='./EN/train', help='Path to training file')
    args = parser.parse_args()
    
    train_file_path = args.train
    
    if args.smoothing:
        print(f"Reading training data from {train_file_path}...")
        train_data = read_train_file(train_file_path)
        print(f"Read {len(train_data)} token-tag pairs.")
        
        # Calculate original vocabulary size
        original_vocab = set(word for word, _ in train_data)
        original_vocab_size = len(original_vocab)
        
        # Estimate emission parameters with smoothing
        print(f"\nEstimating emission parameters WITH smoothing (k={args.k})...")
        emission_params, modified_vocab = estimate_emission_parameters_with_smoothing(train_data, k=args.k)
        
        # Calculate statistics about the smoothing
        modified_vocab_size = len(modified_vocab)
        rare_words_count = original_vocab_size - modified_vocab_size
        
        print(f"\nVocabulary statistics:")
        print(f"Original vocabulary size: {original_vocab_size}")
        print(f"Modified vocabulary size (words appearing >= {args.k} times): {modified_vocab_size}")
        print(f"Number of rare words replaced with #UNK#: {rare_words_count}")
        
    else:
        # For non-smoothing
        print(f"\nEstimating emission parameters WITHOUT smoothing from {train_file_path}...")
        emission_params = estimate_emission_parameters(train_file_path)
        
        train_data = read_train_file(train_file_path)
        original_vocab = set(word for word, _ in train_data)
        original_vocab_size = len(original_vocab)
        print(f"Vocabulary size: {original_vocab_size}")
        
    # Print general statistics
    print(f"\nTotal number of emission parameters: {len(emission_params)}")


if __name__ == "__main__":
    main()
