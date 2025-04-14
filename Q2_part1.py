from Q2_functions import read_training_data, estimate_transition_parameters

def main():
    training_data = read_training_data('EN/train')
    transition_probs = estimate_transition_parameters(training_data)

    # Print the transition probabilities
    for (prev_tag, curr_tag), prob in transition_probs.items():
        print(f"P({curr_tag}|{prev_tag}) = {prob:.4f}")

main()