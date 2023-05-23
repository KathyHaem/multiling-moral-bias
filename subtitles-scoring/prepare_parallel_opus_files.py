import argparse
import itertools
import os.path
import re

from typing import List, Tuple

import jieba as jieba
import spacy_udpipe
from tqdm import tqdm


def read_filter_files(file_a: str, file_b: str) -> Tuple[List[List[str]], List[List[str]]]:
    print("Reading files...")
    lines_a = read_file(file_a)
    lines_parallel = zip(lines_a, read_file(file_b))
    control_char_re = get_control_char_regex()

    lines_filtered_a = []
    lines_filtered_b = []
    for line_a, line_b in tqdm(lines_parallel, total=len(lines_a)):
        if not line_a or not line_b:
            print(f"Discarding a pair because one is falsy: '{line_a}', '{line_b}'")
            continue
        lines_filtered_a.append(remove_control_chars(control_char_re, line_a))
        lines_filtered_b.append(remove_control_chars(control_char_re, line_b))

    print("Finished initial filtering step")
    return lines_filtered_a, lines_filtered_b


def tokenise_parallel_files(lang_a, lang_b, lines_filtered_a, lines_filtered_b):
    lines_tok_a = pretokenise(lang_a, lines_filtered_a)
    lines_tok_b = pretokenise(lang_b, lines_filtered_b)
    # lines_parallel_tok = zip(lines_tok_a, lines_tok_b)
    return lines_tok_a, lines_tok_b


def get_control_char_regex():
    control_chars = ''.join(
        map(chr, itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))))
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    return control_char_re


def remove_control_chars(control_char_re, s):
    return control_char_re.sub('', s)


def pretokenise(lang: str, lines: List[str]) -> List[List[str]]:
    """ Copied and adapted from a previous project:
    actual tokenisation, depending on lang """
    result = []
    if "zh" in lang:
        for line in tqdm(lines, desc="tokenizing with jieba", total=len(lines)):
            tokens = list(jieba.cut(line, cut_all=False))
            result.append(tokens)
        return result

    spacy_udpipe.download(lang)
    nlp = spacy_udpipe.load(lang)
    udpipe_result = nlp.pipe(lines, n_process=32)
    result = []
    try:
        for doc in tqdm(udpipe_result, desc="segmenting and tokenizing with udpipe", total=len(lines)):
            segmented = []
            for sent in doc.sents:
                tokens = [t.text for t in sent]
                segmented.extend(tokens)
            result.append(segmented)
    except ValueError as e:
        print(e)
        print("exiting tokenise due to an error")
    return result


def read_file(filename: str):
    lines = []
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            lines.append(line)
    return lines


def write_file(tok_filename: str, tok_lines: List[List[str]]):
    with open(tok_filename, "w+", encoding="utf-8") as outfile:
        for line in tok_lines:
            outfile.write(" ".join(line))
            outfile.write("\n")


def filter_line_length(lines_tok_a, lines_tok_b, max_tokens):
    lines_parallel_tok = zip(lines_tok_a, lines_tok_b)
    if max_tokens > 0:
        lines_parallel_tok = filter(lambda x: len(x[0]) <= max_tokens and len(x[1]) <= max_tokens, lines_parallel_tok)
    return list(zip(*lines_parallel_tok))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_line_tokens", type=int, help="Filter line length", default=40)
    parser.add_argument("--in_folder", type=str, help="Location of parallel plain text files")
    parser.add_argument("--out_folder", type=str, help="Where to save tokenised and filtered text files")
    parser.add_argument("--lang1", type=str)
    parser.add_argument("--lang2", type=str)
    parser.add_argument("--corpus_name", type=str, default="OpenSubtitles",
                        help="Corpus name as it is part of the file names")
    args = parser.parse_args()

    lang_a = args.lang1
    lang_b = args.lang2
    filename1 = f"{args.corpus_name}.{lang_a}-{lang_b}.{lang_a}"
    filename2 = f"{args.corpus_name}.{lang_a}-{lang_b}.{lang_b}"
    file_path1 = os.path.join(args.in_folder, filename1)
    file_path2 = os.path.join(args.in_folder, filename2)

    lines_filtered_a, lines_filtered_b = read_filter_files(file_path1, file_path2)
    lines_tok_a, lines_tok_b = tokenise_parallel_files(lang_a, lang_b, lines_filtered_a, lines_filtered_b)
    lines_filtered_a, lines_filtered_b = filter_line_length(lines_tok_a, lines_tok_b, max_tokens=args.max_line_tokens)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder, exist_ok=True)
    write_file(tok_filename=os.path.join(args.out_folder, filename1 + ".tok"), tok_lines=lines_filtered_a)
    write_file(tok_filename=os.path.join(args.out_folder, filename2 + ".tok"), tok_lines=lines_filtered_b)


if __name__ == "__main__":
    main()
