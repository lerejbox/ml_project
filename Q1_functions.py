def estimate_emission_parameters(training_data):
    """
    Estimate emission parameters e(x|y) using MLE.
    
    Args:
        training_file_path: Path to the training file
    
    Returns:
        Dictionary mapping (tag, word) pairs to emission probabilities
    """
    tag_word_counts = {}
    tag_counts = {}
    
    # Read and parse the training file
    with open(training_data, 'r', encoding='utf-8') as f:
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
                
                # Increment tag count
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Increment (tag, word) pair count
                tag_word_counts[(tag, word)] = tag_word_counts.get((tag, word), 0) + 1
    
    # Calculate emission probabilities
    emissions = {}
    for (tag, word), count in tag_word_counts.items():
        emissions[(tag, word)] = count / tag_counts[tag]
    
    return emissions



def estimate_emission_parameters_with_smoothing(training_data, k=3):
    """
    Estimate emission parameters e(x|y) using MLE with smoothing for rare words.
    
    Args:
        training_data: List of (word, tag) pairs from the training set
        k: Threshold for rare words (words appearing less than k times will be replaced with #UNK#)
    
    Returns:
        Dictionary mapping (tag, word) pairs to emission probabilities,
        Set of words in the modified vocabulary
    """
    # Count word occurrences to identify rare words
    word_counts = {}
    for word, _ in training_data:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Create modified training data by replacing rare words with #UNK#
    modified_training_data = []
    modified_vocab = set()
    
    for word, tag in training_data:
        if word_counts[word] < k:
            modified_training_data.append(("#UNK#", tag))
        else:
            modified_training_data.append((word, tag))
            modified_vocab.add(word)
    
    # Count tag-word pairs and tags in the modified training data
    tag_word_counts = {}
    tag_counts = {}
    
    for word, tag in modified_training_data:
        # Increment tag count
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Increment (tag, word) pair count
        tag_word_counts[(tag, word)] = tag_word_counts.get((tag, word), 0) + 1
    
    # Calculate emission probabilities
    emissions = {}
    for (tag, word), count in tag_word_counts.items():
        emissions[(tag, word)] = count / tag_counts[tag]
    
    return emissions, modified_vocab


def predict_tags(emission_params, dev_in_path, output_path):
    """
    Predict tags for each word in the sequence using argmax of emission probabilities.
    """
    # Get all possible tags from emission parameters
    all_tags = set(tag for (tag, _) in emission_params.keys())
    
    with open(dev_in_path, 'r', encoding='utf-8') as dev_in, open(output_path, 'w', encoding='utf-8') as dev_out:
        for line in dev_in:
            line = line.strip()
            if not line:
                dev_out.write('\n')
                continue
                
            word = line
            
            # Predict the tag with the highest emission probability
            best_tag = None
            best_prob = -1
            
            for tag in all_tags:
                if (tag, word) in emission_params:
                    prob = emission_params[(tag, word)]
                    if prob > best_prob:
                        best_tag = tag
                        best_prob = prob
            
            if best_tag is None:
                best_tag = 'O'  # Or replace with your most common tag

            dev_out.write(f"{word} {best_tag}\n")