import sys


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
            if len(parts) >= 2:
                words.append(parts[0])
                tags.append(parts[1])

    return words, tags


def calculate_metrics(true_tags, pred_tags):
    """
    Calculate overall precision, recall, and F1 score

    Args:
        true_tags: List of true tags
        pred_tags: List of predicted tags

    Returns:
        Precision, recall, and F1 score
    """
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0

    for true, pred in zip(true_tags, pred_tags):
        if true == pred:
            tp += 1
        else:
            if pred != "O":  # If predicted a tag but got it wrong
                fp += 1
            if true != "O":  # If missed a true tag
                fn += 1

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return precision, recall, f1


# def main():
#     """Main function"""
#     if len(sys.argv) != 3:
#         print("Usage: python simple_metrics.py <true_tags_file> <pred_tags_file>")
#         sys.exit(1)

#     true_file = sys.argv[1]
#     pred_file = sys.argv[2]

#     # Read files
#     _, true_tags = read_file(true_file)
#     _, pred_tags = read_file(pred_file)

#     # Calculate metrics
#     precision, recall, f1 = calculate_metrics(true_tags, pred_tags)

#     # Print results
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")


# if __name__ == "__main__":
#     main()
