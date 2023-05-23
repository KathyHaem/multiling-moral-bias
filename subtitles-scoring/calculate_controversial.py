import argparse
import csv
from typing import List, Tuple

import numpy as np


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


def calculate_mean_std_diff(scores: List[Tuple[str, str, float, float]]) -> Tuple[float, float]:
    """ Takes the statements and scores as loaded from file, calculates the score difference for every matched statement
    and finds the mean absolute difference in scores, as well as the standard deviation. """
    diffs = diffs_list(scores)
    mean = np.mean(diffs).item()
    std = np.std(diffs).item()
    return mean, std


def find_controversial_statements(
        scores: List[Tuple[str, str, float, float]], top_n: int) -> List[Tuple[str, str, float]]:
    """ Takes the statements and scores as loaded from file. Returns a list of statements with larger absolute score
     differences than (mean + some threshold) (for now, 1 standard deviation).
     Return list includes statements from both input files, plus the score difference. """
    results = []
    mean, std = calculate_mean_std_diff(scores)
    for elem in scores:
        diff = elem[3] - elem[2]
        abs_diff = abs(diff)
        if abs_diff - mean > std:
            results.append((elem[0], elem[1], diff))
    results = sorted(results, key=lambda x: abs(x[2]), reverse=True)
    if top_n:
        results = results[:top_n]
    return results


def diffs_list(scores: List[Tuple[str, str, float, float]]) -> List[float]:
    """ Helper function. """
    diffs = [abs(elem[2] - elem[3]) for elem in scores]
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, required=True, help="file path 1")
    parser.add_argument("--file2", type=str, required=True, help="file path 2")
    parser.add_argument("--expect_identical", type=bool,
                        help="If true, expect the statements (first col) in both files to be exactly identical.")
    parser.add_argument("--top_n", type=int, help="Print the top n 'most controversial' statements")
    parser.add_argument("--out_file", type=str, required=False, help="output file")
    args = parser.parse_args()
    scores = load_scored_statements(args.file1, args.file2, args.expect_identical)
    results = find_controversial_statements(scores, args.top_n)
    with open(args.out_file, "w+", newline="") as out_file:
        writer = csv.writer(out_file)
        for row in results:
            writer.writerow(row)


if __name__ == '__main__':
    main()
