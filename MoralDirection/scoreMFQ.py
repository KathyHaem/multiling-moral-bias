import argparse

from mort.funcs_mcm import BERTSentenceSubspace
from tqdm import tqdm
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-large-nli-mean-tokens", help="Model name")
parser.add_argument('--data_cluster', default="atomic", type=str,
                    help='data name')
parser.add_argument('--data', default="context", type=str,
                    help='data name')
parser.add_argument('--cluster', default=2, type=int,
                    help='num cluster')
parser.add_argument('--dim', default=5, type=int,
                    help='dimension of embedding')
parser.add_argument("--batch_size", default=8, type=int, help="No. sentences processed in parallel")
parser.add_argument("--postproc", default=None, type=str,
                    help="Post-processing function that was applied. "
                         "Options: standardize / rm_mean / rm_top5_dims / rm_top_pcs / None")
parser.add_argument("--mfq", default=12, type=int, choices=[1, 2, 12], help="Which MFQ questionnaire to use. "
                                                                            "'12' refers to the rephrased version 1")
parser.add_argument("--lang", default="en", type=str)
parser.add_argument("--norm", type=str, default=None, help="If you really need to pass in a specific norm")
args = parser.parse_args()

projection_file = 'mort/results/bias/cluster/{}_{}_{}Verbs_bertsentenceEmbedding_{}_Cluster{}_Embdim{}_PCA.p'.format(
    args.postproc,
    args.lang,
    args.data_cluster,
    args.model_name,
    args.cluster,
    args.dim
)

questions = []
with open(f"data/mfq-{args.mfq}.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    questions = list(reader)


def compute_bias(bias_func, model_name, name, questions, questions_keys, mfq, lang="en", postproc=None):
    scored_questions = dict()
    for questions_key, question in tqdm(zip(questions_keys, questions)):
        bias_with_q = bias_func({lang: question[lang]})
        if not bias_with_q:
            continue
        scored_questions[questions_key] = bias_with_q[0][0]

    csv_columns = ['Question Number', 'Question', 'Score']

    csv_file = "data/correlation/mfq2/{}_{}_{}_{}_bias_mfq{}.csv".format(
        postproc, lang, name, model_name, mfq)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    try:
        with open(csv_file, 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for question_key in questions_keys:
                if question_key not in scored_questions:
                    continue
                data_row = [
                    question_key,
                    questions[question_key][lang],
                    scored_questions[question_key]
                ]
                writer.writerow(data_row)
        print("Done with " + name)
    except IOError:
        print("I/O error")


if args.norm is not None:
    norm = float(args.norm)
else:
    norm = None
emb_sub = BERTSentenceSubspace(transformer_model=args.model_name, filename_pickled_cluster=projection_file, norm=norm)
questions_keys = range(len(questions))

compute_bias(lambda x: emb_sub.bias_one_phrase_qs_ready(x, use_qa=False),
             args.model_name,
             'BERT_subspace_qa',
             questions, questions_keys,
             args.mfq,
             args.lang,
             args.postproc)

del emb_sub
