from Q1_functions import estimate_emission_parameters_with_smoothing
from Q2_functions import read_training_data, read_data, viterbi, read_training_pairs,estimate_transition_parameters, collect_tags

def main():
    # Getting the transition probability, emission probability and tags from earlier parts
    training_data = read_training_data('EN/train')
    transition_probs = estimate_transition_parameters(training_data)    
    sentences = read_data("EN/dev.in")
    tags, entity_tags = collect_tags(training_data)
    emission_probs, modified_vocab=estimate_emission_parameters_with_smoothing(read_training_pairs("EN/train"))
    
    best_tags=[]

    for sentence in sentences:
        best_tags.append(viterbi(transition_probs,emission_probs,sentence,tags))
    
    # Fix the output formatting section
    formatted_sentences = []
    for i in range(len(sentences)):
        formatted_sentence = []
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            tag = best_tags[i][j]
            formatted_word_tag = f"{word} {tag}"
            formatted_sentence.append(formatted_word_tag)
        formatted_sentences.append(formatted_sentence)

    # Write results to file
    with open('EN\dev.p2.out', 'w', encoding='utf-8') as f:
        for sentence in formatted_sentences:
            for word_tag in sentence:
                f.write(word_tag + '\n')
            f.write('\n')

main()
