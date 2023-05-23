import argparse

from mort.dataMoral import get_translated_qs
from mort.funcs_mcm import BERTSentence, BERTSentenceSubspace
from tqdm import tqdm
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-large-nli-mean-tokens", help="Model name")
parser.add_argument('--data_cluster', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--data', default=None, type=str,
                    help='data name', required=True)
parser.add_argument('--cluster', default=None, type=int,
                    help='num cluster', required=True)
parser.add_argument('--dim', default=None, type=int,
                    help='dimension of embedding', required=True)
parser.add_argument("--batch_size", default=8, type=int, help="No. sentences processed in parallel")
parser.add_argument("--postproc", default=None, type=str,
                    help="Post-processing function that was applied. "
                         "Options: standardize / rm_mean / rm_top5_dims / rm_top_pcs / None")
parser.add_argument('--user_study', default="regional", type=str,
                    help='regional|globalAMT')
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

""" This file seems to be what I need to prep for the user study / correlation thing?? """
with open(f"data/user_study/userStudy_yes_no_{args.user_study}_fixed.csv", "r") as f:
    actions_keys_ = [row.split(',')[0] for row in f]
    actions_keys_ = actions_keys_[1:]

questions = dict()
with open(f"data/user_study/userStudy_action_template_{args.user_study}_fixed.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        questions[row[0]] = row[1]


def compute_bias(bias_func, model_name, name, actions, actions_keys, lang="en", postproc=None, user_study="regional"):
    data_user_study_bert = dict()
    for actions_key, action in tqdm(zip(actions_keys, actions)):
        # bias_with_action, _, _ = bias_func(action)
        bias_with_action = bias_func(action)
        if not bias_with_action:
            continue
        data_user_study_bert[actions_key] = bias_with_action[0][0]

    csv_columns = ['Action', 'Score']

    csv_file = "data/correlation/userstudy/{}_{}_{}_{}_bias_{}.csv".format(
        postproc, lang, name, model_name, user_study)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)
            for action in actions_keys:
                if action not in data_user_study_bert:
                    continue
                data_row = [
                    action,
                    data_user_study_bert[action]
                ]
                writer.writerow(data_row)
        print("Done with " + name)
    except IOError:
        print("I/O error")


# emb = BERTSentence(transformer_model='average_word_embeddings_glove.840B.300d')
# actions_ = actions_keys_.copy()
# compute_bias(lambda x: emb.bias(x), 'glove_cossim', actions_, actions_)
# del emb

#emb = BERTSentence(transformer_model=args.model_name)
#actions_ = actions_keys_.copy()
#if args.lang == "en":
#    compute_bias(lambda x: emb.bias(x), args.model_name, 'BERT_cossim', actions_, actions_, postproc=args.postproc,
#                 user_study=args.user_study)
#else:
#    compute_bias(lambda x: emb.bias_data_ready(get_translated_qs(args.lang, [x])), args.model_name,
#                 'BERT_cossim', actions_, actions_,
#                 args.lang, args.postproc, args.user_study)
#del emb

if args.norm is not None:
    norm = float(args.norm)
else:
    norm = None
emb_sub = BERTSentenceSubspace(transformer_model=args.model_name, filename_pickled_cluster=projection_file, norm=norm)
actions_ = actions_keys_.copy()
if args.lang == "en":
    compute_bias(lambda x: emb_sub.bias(x, qa_template=True), args.model_name,
                 'BERT_subspace_qa', actions_, actions_,
                 postproc=args.postproc, user_study=args.user_study)
else:
    compute_bias(lambda x: emb_sub.bias_one_phrase_qs_ready(get_translated_qs(args.lang, [x])),
                 args.model_name,
                 'BERT_subspace_qa',
                 actions_, actions_,
                 args.lang,
                 args.postproc,
                 args.user_study)
# compute_bias(lambda x: emb_sub.bias(x, qa_template=True), 'BERT_subspace_qa', actions_, actions_, args.lang)

#actions_ = [questions[a] + ' {}'.format(a) for a in actions_keys_]
#if args.lang == "en":
#    compute_bias(lambda x: emb_sub.bias(x, qa_template=False), args.model_name,
#                 'BERT_subspace_raw', actions_, actions_,
#                 postproc=args.postproc, user_study=args.user_study)
# else:
#     compute_bias(lambda x: emb_sub.bias_data_ready(get_translated_qs(args.lang, [x])),
#                  'BERT_subspace_raw',
#                  actions_, actions_,
#                  args.lang)
# compute_bias(lambda x: emb_sub.bias(x, qa_template=False), 'BERT_subspace_raw', actions_, actions_keys_, args.lang)
del emb_sub
