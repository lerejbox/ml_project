import math
from collections import defaultdict

from Q1_functions import (
    estimate_emission_parameters_with_smoothing,
)
from Q1_part1 import read_train_file
from Q2_functions import estimate_transition_parameters, read_training_data

# print(emission_params)


# print(transition_probs)


def evaluate_ner_predictions(gold_file, pred_file):
    """Proper evaluation function for NER with BIO tagging."""

    def extract_entities(file_path):
        """Extract entities from a file with BIO tagging."""
        entities = []
        current_entity = None
        current_tokens = []
        token_index = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_entity:
                        entities.append((current_entity, tuple(current_tokens)))
                        current_entity = None
                        current_tokens = []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]

                    if tag.startswith("B-"):
                        if current_entity:
                            entities.append((current_entity, tuple(current_tokens)))
                        current_entity = tag[2:]  # Remove 'B-' prefix
                        current_tokens = [token_index]
                    elif tag.startswith("I-") and current_entity == tag[2:]:
                        current_tokens.append(token_index)
                    else:
                        if current_entity:
                            entities.append((current_entity, tuple(current_tokens)))
                            current_entity = None
                            current_tokens = []

                    token_index += 1

            if current_entity:
                entities.append((current_entity, tuple(current_tokens)))

        return entities

    # Extract entities from both files
    gold_entities = extract_entities(gold_file)
    pred_entities = extract_entities(pred_file)

    # Convert to sets for comparison
    gold_set = set(gold_entities)
    pred_set = set(pred_entities)

    # Calculate metrics
    correct = len(gold_set.intersection(pred_set))

    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Total gold entities: {len(gold_set)}")
    print(f"Total predicted entities: {len(pred_set)}")
    print(f"Correct predictions: {correct}")

    return precision, recall, f1


def k_best_viterbi_final(tokens, emission_params, transition_probs, k=4):
    """K-best Viterbi implementation with proper smoothing."""

    # Extract all tags
    all_tags = set()
    for t1, t2 in transition_probs.keys():
        if t1 != "START" and t1 != "STOP":
            all_tags.add(t1)
        if t2 != "STOP":
            all_tags.add(t2)
    all_tags = sorted(list(all_tags))

    # Calculate tag frequencies for smoothing
    tag_counts = defaultdict(int)
    total_words = 0
    for (tag, word), prob in emission_params.items():
        tag_counts[tag] += 1
        total_words += 1

    tag_frequencies = {}
    for tag, count in tag_counts.items():
        tag_frequencies[tag] = count / total_words

    # Smoothing function for unknown words
    def get_emission_prob(tag, word):
        if (tag, word) in emission_params:
            return emission_params[(tag, word)]
        else:
            # Frequency-based smoothing
            if tag in tag_frequencies:
                # Higher probability for more frequent tags
                base_prob = 1e-5
                return base_prob * (1 + 10 * tag_frequencies[tag])
            else:
                return 1e-7

    n = len(tokens)
    if n == 0:
        return []

    # Initialize k-best data structure
    dp = []
    for _ in range(n):
        dp.append({tag: [] for tag in all_tags})

    # Initialize first position
    for tag in all_tags:
        trans_prob = transition_probs.get(("START", tag), 1e-10)
        emit_prob = get_emission_prob(tag, tokens[0])

        if trans_prob > 0 and emit_prob > 0:
            score = math.log(trans_prob) + math.log(emit_prob)
            dp[0][tag].append((score, "START", 0))

    # Forward pass
    for j in range(1, n):
        for curr_tag in all_tags:
            candidates = []

            for prev_tag in all_tags:
                for rank, (prev_score, _, _) in enumerate(dp[j - 1][prev_tag]):
                    trans_prob = transition_probs.get((prev_tag, curr_tag), 1e-10)
                    emit_prob = get_emission_prob(curr_tag, tokens[j])

                    if trans_prob > 0 and emit_prob > 0:
                        new_score = (
                            prev_score + math.log(trans_prob) + math.log(emit_prob)
                        )
                        candidates.append((new_score, prev_tag, rank))

            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                dp[j][curr_tag] = candidates[:k]

    # Find final candidates
    final_candidates = []
    for tag in all_tags:
        for rank, (score, _, _) in enumerate(dp[n - 1][tag]):
            trans_prob = transition_probs.get((tag, "STOP"), 1e-10)
            if trans_prob > 0:
                final_score = score + math.log(trans_prob)
                final_candidates.append((final_score, tag, rank))

    if not final_candidates:
        # Fallback
        most_frequent_tag = max(tag_counts.items(), key=lambda x: x[1])[0]
        return [most_frequent_tag] * n

    # Select k-th best
    final_candidates.sort(reverse=True, key=lambda x: x[0])
    selected_index = min(k - 1, len(final_candidates) - 1)

    # Backtrack
    path = []
    curr_tag = final_candidates[selected_index][1]
    curr_rank = final_candidates[selected_index][2]

    for j in range(n - 1, -1, -1):
        path.append(curr_tag)
        if j > 0 and curr_rank < len(dp[j][curr_tag]):
            _, prev_tag, prev_rank = dp[j][curr_tag][curr_rank]
            curr_tag = prev_tag
            curr_rank = prev_rank

    return path[::-1]


def process_file_final(input_file, output_file, emission_params, transition_probs, k=4):
    """Process file with k-best Viterbi."""

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    predictions = []
    current_sentence = []

    for line in lines:
        if not line:
            if current_sentence:
                if k == 1:
                    # Use standard Viterbi for k=1
                    tags = k_best_viterbi_final(
                        current_sentence, emission_params, transition_probs, k=1
                    )
                else:
                    tags = k_best_viterbi_final(
                        current_sentence, emission_params, transition_probs, k
                    )

                for word, tag in zip(current_sentence, tags):
                    predictions.append(f"{word} {tag}")
                predictions.append("")
            current_sentence = []
        else:
            current_sentence.append(line)

    if current_sentence:
        if k == 1:
            tags = k_best_viterbi_final(
                current_sentence, emission_params, transition_probs, k=1
            )
        else:
            tags = k_best_viterbi_final(
                current_sentence, emission_params, transition_probs, k
            )

        for word, tag in zip(current_sentence, tags):
            predictions.append(f"{word} {tag}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))


def final_test(input_file, emission_params, transition_probs):
    # Test with k=4
    print("\nTesting with k=4 (4th best sequence):")
    process_file_final(
        input_file, "EN/dev.p3.out", emission_params, transition_probs, k=4
    )
    precision, recall, f1 = evaluate_ner_predictions("EN/dev.out", "EN/dev.p3.out")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# Main execution
if __name__ == "__main__":

    training_data = read_train_file("EN/train")
    emission_params, modified_vocab = estimate_emission_parameters_with_smoothing(
        training_data, 3
    )

    training_data = read_training_data("EN/train")
    transition_params = estimate_transition_parameters(training_data)
    final_test("EN/dev.in", emission_params, transition_params)
