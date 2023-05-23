"""
/data/correlation/userstudy/userstudyVerbs_use_hubEmbedding.csv
/data/correlation/userstudy/BERT_cossim_bias.csv
/data/correlation/userstudy/BERT_subspace_qa_bias.csv
/data/correlation/userstudy/BERT_subspace_raw_bias.csv
/data/correlation/userstudy/userStudy_scores_regional.csv
"""
import argparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import csv
import os
from matplotlib import rc
import seaborn as sns

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
sns.set(style='ticks', palette='Set2')

rc('text', usetex=True)


def read_bias_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    res = [[float(user_score), float(bert_score), float(mort_score), action] for
           (action, user_score, bert_score, mort_score) in data]
    return res


def own_plot(x, y, a=None, b=None, suffix="", text_pos=(-0.3, 1.2)):
    fontsize = 9
    x_all = x + a
    y_all = y + b

    fig = plt.figure(figsize=(4, 1.4))
    ax = plt.gca()

    plt.scatter(x, y, s=5, color='#BE6F00', label='Do')
    plt.scatter(a, b, s=5, color='#00715E', label='Dont')
    plt.plot(np.unique(x_all), np.poly1d(np.polyfit(x_all, y_all, 1))(np.unique(x_all)),
             label='Correlation', color='#004E8A', gid='r = ' + str(round(pearsonr(x_all, y_all)[0], 3)))
    plt.ylim((-1.1, 1.5))
    plt.yticks(np.arange(-1, 1.1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
    ax.tick_params(axis='both', which='minor', labelsize=fontsize, direction='in')
    r = pearsonr(x_all, y_all)
    asterisks = ''
    if r[1] < 0.05:
        if r[1] < 0.01:
            if r[1] < 0.001:
                asterisks = '***'
            else:
                asterisks = '**'
        else:
            asterisks = '*'
    print(r)
    # input("Press key")
    plt.xlabel('MCM score', fontsize=fontsize - 1)
    plt.ylabel('User Study value', fontsize=fontsize - 1)
    # plt.tight_layout()
    # plt.text(-0.8, 0.12, 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A', fontsize=10)
    if "BERT_stsb_cossim" in suffix:
        plt.title("\\textbf{BERT$_{stsb}$ (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
    elif "BERT_cossim" in suffix:
        plt.title("\\textbf{BERT (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
    elif "BERT_subspace_qa" in suffix:
        plt.title("\\textbf{BERT (Moral Compass QT)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
        # plt.xlim((-1.1, 1.1))
    elif "BERT_subspace_raw" in suffix:
        plt.title("\\textbf{BERT (Moral Compass)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
        plt.xticks(np.arange(-1, 1.1, 0.25))
        # plt.xlim((-1.1, 1.1))
    elif "glove" in suffix:
        plt.title("\\textbf{GloVe (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
    else:
        plt.title("\\textbf{USE (Cosine Similarity)}", fontsize=fontsize)
        plt.text(text_pos[0], text_pos[1], 'r = ' + str(round(r[0], 2)) + asterisks, color='#004E8A',
                 fontsize=fontsize - 1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.grid(True, linestyle=':')
    os.makedirs('mort/plot_corr/userstudy/plots/', exist_ok=True)
    svg_out = 'mort/plot_corr/userstudy/plots/correlation_{}.svg'.format(suffix)
    svg_dir = os.path.split(svg_out)[0]
    os.makedirs(svg_dir, exist_ok=True)
    plt.savefig(svg_out, bbox_inches='tight', dpi=600)
    # plt.show()
    plt.clf()
    plt.close()
    # exit()


# input("Press key")
def _corr(args, lang_a, lang_b, data_dos, data_donts, model_name, text_pos, user_study: str):

    x = [b[2] for b in data_dos]  # user dos
    y = [[b[0], b[1]] for b in data_dos]  # action + mcm does
    a = [b[2] for b in data_donts]  # user donts
    b = [[b[0], b[1]] for b in data_donts]  # action + mcm donts

    """for elem in y:
        print(elem)
    for elem in b:
        print(elem)"""

    y = [p[1] for p in y]  # mcm does
    b = [p[1] for p in b]  # mcm donts

    own_plot(x, y, a, b, suffix="{}_{}_vs_{}_{}userstudy_{}_{}".format(
        args.postproc, lang_a, lang_b, user_study, model_name, args.model_name), text_pos=text_pos)

    f = np.array(x + a)
    d = np.array(y + b)
    # print('###', pearsonr(x, y))
    print('Pearson ###', pearsonr(f, d))


def sort_dos_donts(data_mcm, dim_action, dim_score, data_user):
    # sort mcm data and weat
    actions = set([x[0] for x in data_mcm]) & set([x[0] for x in data_user])
    data_user = [x for x in data_user if x[0] in actions]
    data_mcm = [x for x in data_mcm if x[0] in actions]
    data_user.sort(key=lambda x: x[0])
    data_mcm.sort(key=lambda x: x[dim_action])
    assert len(data_user) == len(data_mcm)
    for d_m, d_w in zip(data_mcm, data_user):
        assert d_m[dim_action] == d_w[0]
    data_dos = [[d_w[0], float(d_w[1]), float(d_m[dim_score])] for (d_m, d_w) in zip(data_mcm, data_user)]
    return data_dos


def run_correlations(args, user_study):
    with open(f"data/correlation/userstudy/userStudy_scores_{user_study}.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    data = [[action, (float(user_score) - 0.5) / 0.5] for (action, user_score) in data]
    # separate dos and donts based on user study scores
    dos_study = [[action, float(user_score)] for (action, user_score) in data if float(user_score) >= 0.]
    donts_study = [[action, float(user_score)] for (action, user_score) in data if float(user_score) < 0.]
    text_pos = (-0.25, 1.2,)

    lang_a = "zh"
    model_csv_file = "data/correlation/userstudy/{}_{}_{}_{}_bias_{}.csv".format(
        args.postproc, lang_a, args.mode, args.model_name, user_study)
    # read dos based on input does
    with open(model_csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        d = list(reader)
        try:
            tmp = float(d[-1][0])
            dim_score = 0
            dim_action = 1
        except:
            dim_score = 1
            dim_action = 0
        data_mcm = [x for x in d if x[dim_action] in list(zip(*dos_study))[0]]

    data_dos_a = sort_dos_donts(data_mcm, dim_action, dim_score, dos_study)

    # read donts based on input donts
    with open(model_csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        d = list(reader)
        data_mcm = [x for x in d if x[dim_action] in list(zip(*donts_study))[0]]

    data_donts_a = sort_dos_donts(data_mcm, dim_action, dim_score, donts_study)

    for lang_b in args.lang.split():
        model_csv_file = "data/correlation/userstudy/{}_{}_{}_{}_bias_{}.csv".format(
            args.postproc, lang_b, args.mode, args.model_name, user_study)
        # read dos based on input does
        with open(model_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            d = list(reader)
            try:
                tmp = float(d[-1][0])
                dim_score = 0
                dim_action = 1
            except:
                dim_score = 1
                dim_action = 0
            data_mcm = [x for x in d if x[dim_action] in list(zip(*dos_study))[0]]

        data_dos = sort_dos_donts(data_mcm, dim_action, dim_score, dos_study)
        data_dos = [[d_b[0], float(d_b[2]), float(d_a[2])] for (d_b, d_a) in zip(data_dos, data_dos_a)]

        # read donts based on input donts
        with open(model_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            d = list(reader)
            data_mcm = [x for x in d if x[dim_action] in list(zip(*donts_study))[0]]

        data_donts = sort_dos_donts(data_mcm, dim_action, dim_score, donts_study)
        data_donts = [[d_b[0], float(d_b[2]), float(d_a[2])] for (d_b, d_a) in zip(data_donts, data_donts_a)]

        _corr(args, lang_a, lang_b, data_dos, data_donts, args.mode, text_pos, user_study)


def parse_args():
    parser = argparse.ArgumentParser(description='User study plots')
    parser.add_argument('--user_study', default="globalAMT", type=str,
                        help='regional|globalAMT|both')
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens", help="Model name")
    parser.add_argument("--postproc", default=None, type=str,
                        help="Post-processing function that was applied. "
                             "Options: standardize / rm_mean / rm_top5_dims / rm_top_pcs / None")
    parser.add_argument("--lang", default="en ar cs de zh", type=str)
    parser.add_argument("--mode", default="BERT_subspace_qa", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    user_study = args.user_study
    if user_study in ["regional", "both"]:
        run_correlations(args, "regional")
    if user_study in ["globalAMT", "both"]:
        run_correlations(args, "globalAMT")


if __name__ == '__main__':
    main()
