import os
import sys

import numpy as np


class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, spectral_radius=0.9):
        # Initialize network parameters
        self.reservoir_size = reservoir_size
        self.input_size = input_size  # Added input_size parameter

        # Reservoir weights
        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

        # Input weights - Adjusted to match input_size
        self.W_in = np.random.rand(reservoir_size, input_size) - 0.5

        # Output weights (to be trained)
        self.W_out = None

    def train(self, input_data, target_data):
        # Run reservoir with input data
        reservoir_states = self.run_reservoir(input_data)

        # Train the output weights using pseudo-inverse
        self.W_out = np.dot(np.linalg.pinv(reservoir_states), target_data)

    def predict(self, input_data):
        # Run reservoir with input data
        reservoir_states = self.run_reservoir(input_data)

        # Make predictions using the trained output weights
        predictions = np.dot(reservoir_states, self.W_out)
        return predictions

    def run_reservoir(self, input_data):
        # Input data should be of shape (time_steps, input_size)
        time_steps = input_data.shape[0]

        # Initialize reservoir states
        reservoir_states = np.zeros((time_steps, self.reservoir_size))

        # Run the reservoir
        for t in range(1, time_steps):
            # Use proper matrix multiplication with the correct input shape
            reservoir_states[t, :] = np.tanh(
                np.dot(self.W_res, reservoir_states[t - 1, :])
                + np.dot(self.W_in, input_data[t, :])
            )

        return reservoir_states


def read_file(filename):
    """Read words and tags from file (one word+tag per line)"""
    words = []
    tags = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 1:
                words.append(parts[0])
                if len(parts) >= 2:
                    tags.append(parts[1])

    return words, tags if tags else None


def extract_features(word):
    """
    Extract features from a word for NER task
    Returns a feature vector as a numpy array
    """
    features = []

    # Is first letter capitalized?
    features.append(1.0 if word and word[0].isupper() else 0.0)

    # Is all uppercase?
    features.append(1.0 if word.isupper() else 0.0)

    # Is all lowercase?
    features.append(1.0 if word.islower() else 0.0)

    # Contains digit?
    features.append(1.0 if any(c.isdigit() for c in word) else 0.0)

    # Contains punctuation?
    features.append(1.0 if any(not c.isalnum() for c in word) else 0.0)

    # Is a number?
    features.append(1.0 if word.replace(".", "").replace(",", "").isdigit() else 0.0)

    # Word length (normalized)
    features.append(min(len(word) / 15.0, 1.0))

    return np.array(features)


def prepare_data(words, tags=None):
    """
    Convert words and (optionally) tags to formats suitable for ESN
    """
    # Extract features for words
    features = np.array([extract_features(word) for word in words])

    # If tags are provided, convert them to one-hot encoding
    if tags:
        unique_tags = sorted(list(set(tags)))
        tag_to_idx = {tag: i for i, tag in enumerate(unique_tags)}

        # One-hot encode tags
        one_hot_tags = np.zeros((len(tags), len(unique_tags)))
        for i, tag in enumerate(tags):
            one_hot_tags[i, tag_to_idx[tag]] = 1

        return features, one_hot_tags, unique_tags

    return features


def ner_with_esn(train_file, dev_in_file, dev_out_file, reservoir_size=100):
    """
    Perform NER using Echo State Network

    Args:
        train_file: Path to training file
        dev_in_file: Path to development input file
        dev_out_file: Path to development output file
        reservoir_size: Size of the ESN reservoir

    Returns:
        Trained ESN, predictions on dev set, and evaluation metrics
    """
    # Read data files
    print(f"Reading training file: {train_file}")
    train_words, train_tags = read_file(train_file)
    print(f"Read {len(train_words)} words from training file")

    print(f"Reading dev input file: {dev_in_file}")
    dev_in_words, _ = read_file(dev_in_file)
    print(f"Read {len(dev_in_words)} words from dev input file")

    print(f"Reading dev output file: {dev_out_file}")
    dev_out_words, dev_out_tags = read_file(dev_out_file)
    print(f"Read {len(dev_out_words)} words with tags from dev output file")

    # Prepare data for ESN
    print("Preparing data...")
    X_train, y_train, unique_tags = prepare_data(train_words, train_tags)
    X_dev = prepare_data(dev_in_words)

    # Get input feature size
    input_size = X_train.shape[1]
    print(f"Input feature size: {input_size}")

    # Initialize ESN with correct input size
    print(
        f"Initializing ESN (reservoir size: {reservoir_size}, input size: {input_size})..."
    )
    esn = EchoStateNetwork(input_size=input_size, reservoir_size=reservoir_size)

    # Train ESN
    print("Training ESN...")
    esn.train(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = esn.predict(X_dev)

    # Convert predictions back to tags
    idx_to_tag = {i: tag for i, tag in enumerate(unique_tags)}
    pred_tags = [idx_to_tag[np.argmax(pred)] for pred in y_pred]

    # Calculate accuracy
    accuracy = sum(p == t for p, t in zip(pred_tags, dev_out_tags)) / len(dev_out_tags)
    print(f"Accuracy: {accuracy:.4f}")

    return esn, pred_tags, accuracy, unique_tags


def main():
    """
    Main function to run NER with ESN
    """
    if len(sys.argv) != 4:
        print("Usage: python esn_ner.py <train_file> <dev_in_file> <dev_out_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    dev_in_file = sys.argv[2]
    dev_out_file = sys.argv[3]

    # Run NER with ESN
    esn, predictions, accuracy, unique_tags = ner_with_esn(
        train_file, dev_in_file, dev_out_file, reservoir_size=300
    )

    # Print results
    print(f"Accuracy on dev set: {accuracy:.2%}")
    print(f"Unique tags: {unique_tags}")

    # Print first 10 predictions
    dev_in_words, _ = read_file(dev_in_file)
    print("\nSample predictions:")
    for i in range(min(10, len(dev_in_words))):
        print(f"{dev_in_words[i]} {predictions[i]}")

    # Save predictions to file
    output_file = "./EN/dev.p4.out"
    with open(output_file, "w", encoding="utf-8") as f:
        for word, tag in zip(dev_in_words, predictions):
            f.write(f"{word} {tag}\n")

    print(f"\nPredictions written to {output_file}")


if __name__ == "__main__":
    main()
