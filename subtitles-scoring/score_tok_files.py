import argparse
import csv
from datetime import datetime
import os

from typing import List, Union, Tuple, Iterable

from BertSentenceSubspace import BERTSentenceSubspace


def compute_bias(
        mcm_model: BERTSentenceSubspace,
        statements: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        batch_size: int = 32
) -> Union[Tuple[Union[str, List[str]], float], Iterable[Tuple[Union[str, List[str]], float]]]:

    if (is_split_into_words and isinstance(statements, str)) \
            or (not is_split_into_words and isinstance(statements, list) and isinstance(statements[0], list)):
        raise ValueError("If is_split_into_words, expect a list of strings or list of lists. "
                         "If not is_split_into_words, expect a string or list of strings.")

    is_batch = False
    if isinstance(statements, list):
        if (not is_split_into_words) or (is_split_into_words and isinstance(statements[0], list)):
            is_batch = True

    if not is_batch:
        score = mcm_model.bias(statements, is_batch=is_batch, pre_tokenised=is_split_into_words)
        return statements, score[0]

    scores = mcm_model.bias(statements, is_batch=is_batch, pre_tokenised=is_split_into_words, batch_size=batch_size)
    statements = [" ".join(stmnt) for stmnt in statements]
    scored_stmnts = zip(statements, scores)
    return scored_stmnts


def save_scores(csv_file, scored_stmnts):
    csv_columns = ['Action', 'Score']
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_new = True
    if os.path.isfile(csv_file):
        file_new = False
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if file_new:
                writer.writerow(csv_columns)
            for data_row in scored_stmnts:
                writer.writerow(data_row)
    except IOError as e:
        print("I/O error " + e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_file", type=str, help="Complete filename of tokenised file to be scored")
    parser.add_argument("--is_split_into_words", type=bool, default=True,
                        help="Assume contents of tok_file have been tokenised and saved with spaces between tokens.")
    parser.add_argument("--out_folder", type=str, help="Where to save scores")
    parser.add_argument("--lang", type=str)
    parser.add_argument("--model_base_folder", type=str, help="Location of MCM model(s)")
    parser.add_argument("--model_name", type=str, default="bert-large-nli-mean-tokens", help="Model name")
    parser.add_argument("--model_base_if_local", type=str, default="../MoRT_NMI/MoRT")
    parser.add_argument('--data_cluster', default="atomic", type=str, help='data name')
    parser.add_argument('--data', default="context", type=str, help='data name')
    parser.add_argument('--cluster', default=2, type=int, help='num cluster')
    parser.add_argument('--dim', default=5, type=int, help='dimension of PCA embedding')
    parser.add_argument("--batch_size", default=64, type=int, help="No. sentences processed in parallel")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use for processing")
    parser.add_argument("--overwrite", default=False, type=bool, help="Do scoring even if out file exists")
    parser.add_argument("--postproc", default=None, type=str,
                        help="Post-processing function that was applied. "
                             "Options: standardize / rm_mean / rm_top5_dims / rm_top_pcs / None")
    args = parser.parse_args()

    out_csv_file = os.path.join(args.out_folder,
                                f"{args.lang}_subspace_qa_{args.model_name}_{os.path.basename(args.tok_file)}.csv")
    if os.path.isfile(out_csv_file) and not args.overwrite:
        print(f"Target file {out_csv_file} already exists. Add --overwrite to score anyway.")
        return

    model_name_local = os.path.join(args.model_base_if_local, args.model_name)
    model_is_local = os.path.isdir(model_name_local)
    lang = "zh" if args.lang == "zh_cn" else args.lang
    projection_filename = f'{args.postproc}_{lang}_' \
                          f'{args.data_cluster}Verbs_bertsentenceEmbedding_{args.model_name}_Cluster{args.cluster}_' \
                          f'Embdim{args.dim}_PCA.p'
    projection_file = os.path.join(args.model_base_folder, projection_filename)

    emb_sub = BERTSentenceSubspace(
        transformer_model=model_name_local if model_is_local else args.model_name,
        filename_pickled_cluster=projection_file, device=args.device
    )

    statements = []
    print("[" + datetime.now().isoformat() + "]" + "loading " + args.tok_file)
    with open(args.tok_file, "r", encoding="utf-8") as tok_file:
        i = 0
        for line in tok_file:
            i += 1
            if args.is_split_into_words:
                statements.append(line.strip().split())
            else:
                statements.append(line.strip())
            if i >= 100_000:  # e.g.
                i = 0
                scored_stmnts = compute_bias(
                    emb_sub, statements, is_split_into_words=args.is_split_into_words, batch_size=args.batch_size)
                statements = []
                save_scores(out_csv_file, scored_stmnts=scored_stmnts)
    scored_stmnts = compute_bias(
        emb_sub, statements, is_split_into_words=args.is_split_into_words, batch_size=args.batch_size)
    save_scores(out_csv_file, scored_stmnts=scored_stmnts)


if __name__ == "__main__":
    main()
