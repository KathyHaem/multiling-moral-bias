import argparse
import csv
from typing import List, Tuple

from scipy.stats import pearsonr


def load_scored_statements(
        file_a: str, file_b: str, expect_identical: bool = False) -> List[Tuple[str, str, float, float]]:
    """ Assumes that each file contains a CSV table with the first column an action or statement, and the second column
    a score. The statements are identical or translations of each other, i.e., they should be in the same order in both
    files. The scores are from different models and may or may not be exactly comparable.
    This function is only responsible for loading the statements and scores!
    """
    with open(file_a, "r", encoding="utf-8") as infile:
        file_a_list = read_scores(infile)
    with open(file_b, "r", encoding="utf-8") as infile:
        file_b_list = read_scores(infile)
    output = [(elem_a[0], elem_b[0], elem_a[1], elem_b[1]) for elem_a, elem_b in zip(file_a_list, file_b_list)]
    if expect_identical and not all([elem[0] == elem[1] for elem in output]):
        print(output)
        raise ValueError("Expected statements to be identical but found a mismatch")
    return output


def read_scores(infile) -> List[Tuple[str, float]]:
    """ Helper function. """
    reader = csv.DictReader(infile, delimiter=',', fieldnames=["statement", "score"])
    scores_list = []
    for row in reader:
        try:
            stmnt = row["statement"]
            score = float(row["score"])
        except ValueError:
            continue
        scores_list.append((stmnt, score))
    return scores_list


def correlate(scores: List[Tuple[str, str, float, float]]) -> Tuple[float, float]:
    x = [elem[2] for elem in scores]
    y = [elem[3] for elem in scores]
    r, p = pearsonr(x, y)
    return r, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, required=True, help="file path 1")
    parser.add_argument("--file2", type=str, required=True, help="file path 2")
    parser.add_argument("--expect_identical", action="store_true",
                        help="If true, expect the statements (first col) in both files to be exactly identical.")
    args = parser.parse_args()
    scores = load_scored_statements(args.file1, args.file2, expect_identical=args.expect_identical)
    results = correlate(scores)
    print(results)


if __name__ == '__main__':
    main()

