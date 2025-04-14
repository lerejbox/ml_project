from collections import defaultdict
from math import log

def read_data(file_path):
    sentences=[]
    current_sentence=[]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                line=line.strip()
                current_sentence.append(line)
                        
        if current_sentence:  # Add last sentence if file doesn't end with newline
            sentences.append(current_sentence)
            
    return sentences

def read_training_pairs(file_path):
    training_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split()
                training_pairs.append((word, tag))
    return training_pairs

def read_training_data(file_path):
    training_data = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                if current_sentence:
                    training_data.append(current_sentence)
                    current_sentence = []
            else:
                word, tag = line.strip().split()
                current_sentence.append(tag)
                
        if current_sentence:  # Add last sentence if file doesn't end with newline
            training_data.append(current_sentence)
            
    return training_data

def estimate_transition_parameters(training_data):
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for sentence in training_data:
        # Add START and STOP tags
        tags = ['START'] + sentence + ['STOP']
        
        for i in range(len(tags) - 1):
            prev_tag = tags[i]
            curr_tag = tags[i + 1]
            
            # Count transitions and individual tags
            transition_counts[(prev_tag, curr_tag)] += 1
            tag_counts[prev_tag] += 1

    # Calculate probabilities with smoothing
    SMOOTH = 1e-5
    transition_probs = {}
    for (prev_tag, curr_tag), count in transition_counts.items():
        transition_probs[(prev_tag, curr_tag)] = (count + SMOOTH) / (tag_counts[prev_tag] + SMOOTH * len(tag_counts))

    return transition_probs

def collect_tags(training_data):
    """Collect all possible tags with special attention to entity tags"""
    tags = set(['START', 'STOP'])
    entity_tags = set()
    for sentence in training_data:
        for tag in sentence:
            tags.add(tag)
            if tag.startswith('B-') or tag.startswith('I-'):
                entity_tags.add(tag)
    return tags, entity_tags

def viterbi(transition_probs,emission_probs,sentence,possible_tags):
    
    # Add small smoothing constant to prevent log(0)
    SMOOTH = 1e-5
    
    # Initialize matrices for storing probabilities and backpointer
    viterbi_matrix = [{}]
    backpointer = [{} for _ in range(len(sentence))]

    # Initialize base case for first word
    first_word = sentence[0]
    for tag in possible_tags:
        # Add smoothing to prevent log(0)
        trans_prob = transition_probs.get(("START", tag), SMOOTH)
        emit_prob = emission_probs.get((first_word, tag), SMOOTH)
        viterbi_matrix[0][tag] = log(trans_prob) + log(emit_prob)
        backpointer[0][tag] = "START"
    
    # Continue for rest of the words
    for t in range(1, len(sentence)):
        viterbi_matrix.append({})
        curr_word = sentence[t]

        for curr_tag in possible_tags:
            max_prob = float('-inf')
            best_prev_tag = None

            # Check all possible previous tags
            for prev_tag in possible_tags:
                prob = (viterbi_matrix[t-1][prev_tag] + 
                       log(transition_probs.get((prev_tag, curr_tag), SMOOTH)) + 
                       log(emission_probs.get((curr_word, curr_tag), SMOOTH)))

                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
                    
            # Store best probability
            viterbi_matrix[t][curr_tag] = max_prob
            backpointer[t][curr_tag] = best_prev_tag

    # Find the best final tag
    final_max_prob = float('-inf')
    best_final_tag = None
    
    for tag in possible_tags:
        prob = viterbi_matrix[-1][tag] + log(transition_probs.get((tag, 'STOP'), SMOOTH))
        if prob > final_max_prob:
            final_max_prob = prob
            best_final_tag = tag
    
    # Back track to find the tags
    best_path = []
    current_tag = best_final_tag
    
    # Backtrack from the last position to the first
    for t in range(len(sentence)-1, -1, -1):
        best_path.insert(0, current_tag)  # Insert at beginning
        current_tag = backpointer[t][current_tag]
    
    return best_path
