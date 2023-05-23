# from "All Bark and No Bite"

import os
import pickle
from glob import glob
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from scipy import stats

import numpy as np


def get_corpus_sample_embs(model_name):
    all_embs_fs = glob('wordsim_embs/{}/*.p'.format(model_name))
    all_mean_embs = []
    for word_fname in all_embs_fs:
        mean_emb = pickle.load(open(word_fname, 'rb'))
        all_mean_embs.append(mean_emb)
    return np.stack(all_mean_embs, axis=1)


# get embedding sample for mean/std of emb space, get PCs for all-but-the-top (Mu et al. 2018)
def get_model_mean_std_pcs(model_name):
    corpus_sample_embeds = get_corpus_sample_embs(model_name)
    corpus_sample_means = corpus_sample_embeds.mean(axis=1)
    corpus_sample_stds = corpus_sample_embeds.std(axis=1)
    num_pcs = 3
    corpus_sample_pcs = []
    for layer in range(corpus_sample_embeds.shape[0]):
        layer_sample = corpus_sample_embeds[layer, :, :]
        layer_sample = layer_sample - corpus_sample_means[layer]
        pca = PCA(n_components=num_pcs)
        pca.fit(layer_sample)
        corpus_sample_pcs.append(pca.components_)
    return corpus_sample_means, corpus_sample_stds, corpus_sample_pcs


def get_sample_means(sample_embeds, axis=0):
    return sample_embeds.mean(axis=axis)


def get_sample_stds(sample_embeds, axis=0):
    return sample_embeds.std(axis=axis)


def get_sample_pcs(sample_embeds, axis=0):
    """ PCs for vectors of shape data x emb dim """
    sample_means = sample_embeds.mean(axis=axis)
    num_pcs = 3
    sample = sample_embeds - sample_means
    pca = PCA(n_components=num_pcs)
    pca.fit(sample)
    sample_pcs = pca.components_
    return sample_pcs


def ustd_cosine(emb1, emb2):
    return 1 - cosine(emb1, emb2)


def dim_rm_cosine(emb1, emb2, mean):
    emb1_rm = emb1.copy()
    emb2_rm = emb2.copy()
    top_5_dims = np.argsort(np.abs(mean))[-5:]
    for dim in top_5_dims:
        emb1_rm[dim] = 0
        emb2_rm[dim] = 0
    return ustd_cosine(emb1_rm, emb2_rm)


def embedding_rm_top5_dim(emb, mean):
    emb_rm = emb.copy()
    top_5_dims = np.argsort(np.abs(mean))[-5:]
    for dim in top_5_dims:
        emb_rm[dim] = 0
    return emb_rm


def spearman(emb1, emb2):
    return stats.spearmanr(emb1, emb2)[0]


def standardised_embeds(emb1, emb2, mean, std):
    mean_rm_emb1, mean_rm_emb2 = mean_rm_embeds(emb1, emb2, mean)
    emb1_std = mean_rm_emb1 / std
    emb2_std = mean_rm_emb2 / std
    return emb1_std, emb2_std


def standardised_embedding(emb, mean, std):
    mean_rm_embed = emb - mean
    return mean_rm_embed / std


def mean_rm_embeds(emb1, emb2, mean):
    """ little helper method for some of the similarity functions """
    mean_rm_emb1 = emb1 - mean
    mean_rm_emb2 = emb2 - mean
    return mean_rm_emb1, mean_rm_emb2


def pcs_rm_embeds(emb1, emb2, mean, pcs):  # = ABTT
    mean_rm_emb1, mean_rm_emb2 = mean_rm_embeds(emb1, emb2, mean)
    rm_term_1 = np.zeros(emb1.shape[0])
    rm_term_2 = np.zeros(emb2.shape[0])
    for pc in pcs:
        rm_term_1 += pc.dot(emb1) * pc
        rm_term_2 += pc.dot(emb2) * pc
    emb1_pc_rm = mean_rm_emb1 - rm_term_1
    emb2_pc_rm = mean_rm_emb2 - rm_term_2
    return emb1_pc_rm, emb2_pc_rm


def pcs_rm_embedding(emb, mean, pcs):  # = ABTT
    mean_rm_emb = emb - mean
    rm_term = np.zeros(emb.shape[0])
    for pc in pcs:
        rm_term += pc.dot(emb) * pc
    return mean_rm_emb - rm_term


def proc_corpus_sample(sample_embeds, axis=0, postproc="standardize", means=None, stds=None):
    means = get_sample_means(sample_embeds, axis) if means is None else means
    if postproc == "rm_mean":
        return sample_embeds - means
    if postproc == "rm_top5_dims":
        emb_rm = sample_embeds.copy()
        top_5_dims = np.argsort(np.abs(means))[-5:]
        for dim in top_5_dims:
            emb_rm[:, dim] = 0
        return emb_rm
    if postproc == "rm_top_pcs":
        pcs = get_sample_pcs(sample_embeds, axis)
        for i, sample in enumerate(sample_embeds):
            sample_embeds[i] = pcs_rm_embedding(sample, means, pcs)
        return sample_embeds
    if postproc == "standardize":
        stds = get_sample_stds(sample_embeds, axis) if stds is None else stds
        return (sample_embeds - means) / stds
    return sample_embeds


def get_decontextualized_sim(word1, word2, model_name, corpus_sample_means, corpus_sample_stds, corpus_sample_pcs):
    if not os.path.exists('wordsim_embs/{}/{}.p'.format(model_name, word1)):
        return None
    if not os.path.exists('wordsim_embs/{}/{}.p'.format(model_name, word2)):
        return None
    word1_emb = pickle.load(open('wordsim_embs/{}/{}.p'.format(model_name, word1), 'rb'))
    word2_emb = pickle.load(open('wordsim_embs/{}/{}.p'.format(model_name, word2), 'rb'))
    layer_sims = {'ustd_cosine': [],
                  'std_cosine': [],
                  'mean_sub_cosine': [],
                  'rm_pcs_cosine': [],
                  'spearman': []}
    for layer in range(word1_emb.shape[0]):
        layer_sims['ustd_cosine'].append(ustd_cosine(word1_emb[layer], word2_emb[layer]))
        layer_sims['spearman'].append(spearman(word1_emb[layer], word2_emb[layer]))
        layer_sims['std_cosine'].append(
            ustd_cosine(*standardised_embeds(word1_emb[layer], word2_emb[layer], mean=corpus_sample_means[layer],
                        std=corpus_sample_stds[layer])))
        layer_sims['mean_sub_cosine'].append(
            ustd_cosine(*mean_rm_embeds(word1_emb[layer], word2_emb[layer], mean=corpus_sample_means[layer])))
        layer_sims['rm_pcs_cosine'].append(
            ustd_cosine(*pcs_rm_embeds(word1_emb[layer], word2_emb[layer],
                                       mean=corpus_sample_means[layer], pcs=corpus_sample_pcs[layer])))

    return layer_sims


def main():
    # load similarity datasets
    sim_filenames = ['word_sim/simlex-999/SimLex-999.txt',
                     'word_sim/simverb-3500/SimVerb-3500.txt',
                     'word_sim/wordsim-353_r/wordsim_relatedness_goldstandard.txt',
                     'word_sim/wordsim-353_s/wordsim_similarity_goldstandard.txt',
                     'word_sim/rg-65/RG65.txt']

    sim_lex_pairs = {}
    for sim_lex_filename in sim_filenames:
        dataset = sim_lex_filename.split('/')[0]
        if dataset not in sim_lex_pairs.keys():
            sim_lex_pairs[dataset] = []
        with open(sim_lex_filename, 'r', encoding='utf-8') as f:
            file_basename = os.path.basename(sim_lex_filename).lower()
            word_pair_lines = f.readlines()
            for line in word_pair_lines[1:]:

                if file_basename[:2] == 'rg':
                    vals = line.split(";")
                else:
                    vals = line.split()
                word1 = vals[0].lower()
                word2 = vals[1].lower()

                if file_basename[:2] == 'wo' or file_basename[:2] == 'rg':
                    score = float(vals[2])
                else:
                    score = float(vals[3])
                sim_lex_pairs[dataset].append((word1, word2, score))

    sim_lex_pairs = {k: list(set(v)) for k, v in sim_lex_pairs.items()}

    # generate the plots!
    models = ['gpt2', 'xlnet-base-cased', 'bert-base-cased', 'roberta-base']
    for model_name in models:
        corpus_sample_means, corpus_sample_stds, corpus_sample_pcs = get_model_mean_std_pcs(model_name)

        excl = 0
        dataset_sims = {}
        for sim_dataset, sim_list in sim_lex_pairs.items():
            layerwise_decontextual_sims = [{'ustd_cosine': [],
                                            'std_cosine': [],
                                            'mean_sub_cosine': [],
                                            'rm_pcs_cosine': [],
                                            'spearman': [],
                                            'human': []} for x in range(13)]

            for sim_pair in sim_list:
                human_score = sim_pair[2]
                word1 = sim_pair[0]
                word2 = sim_pair[1]
                decontextual_sim_score = get_decontextualized_sim(word1, word2, model_name, corpus_sample_means,
                                                                  corpus_sample_stds, corpus_sample_pcs)

                if decontextual_sim_score is not None:
                    for sim_type, layer_scores in decontextual_sim_score.items():
                        for layer_ind, layer_score in enumerate(layer_scores):
                            layerwise_decontextual_sims[layer_ind][sim_type].append(layer_score)
                    for layer_ind in range(13):
                        layerwise_decontextual_sims[layer_ind]['human'].append(human_score)
                else:
                    excl += 1
            dataset_sims[sim_dataset] = layerwise_decontextual_sims

        # generate_plots(model_name, dataset_sims)
        print(excl)


if __name__ == "__main__":
    main()
