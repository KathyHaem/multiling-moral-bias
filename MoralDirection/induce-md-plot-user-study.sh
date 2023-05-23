#!/bin/bash

set -x

# monoling ones
AR_MODELS="aubmindlab/bert-base-arabertv02 kathaem/aubmindlab-arabertv02-base-sentence-transformer-xnli-ar"
CS_MODELS="ufal/robeczech-base UWB-AIR/Czert-B-base-cased kathaem/ufal-robeczech-base-sentence-transformer-mnli-cs"
DE_MODELS="deepset/gbert-base bert-base-german-cased kathaem/deepset-gbert-base-sentence-transformer-xnli-de"
EN_MODELS="bert-base-cased bert-large-cased bert-large-nli-mean-tokens kathaem/bert-base-cased-sentence-transformer-mnli-en"
ZH_MODELS="bert-base-chinese output/bert-base-chinese_nliv2_zh kathaem/bert-base-chinese-sentence-transformer-xnli-zh"

MULTILING_MODELS="xlm-roberta-base bert-base-multilingual-cased kathaem/xlm-roberta-base-sentence-transformer-nli-5langs sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens"

for LANG in ar cs de en zh; do
  for MODEL in "${MULTILING_MODELS[@]}"; do
    python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
        --lang ${LANG} --bert_model_name "${MODEL}"

    python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
        --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

    python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
  done
done

LANG="ar"
for MODEL in "${AR_MODELS[@]}"; do
  python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
    --lang ${LANG} --bert_model_name "${MODEL}"

  python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
    --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

  python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
done

LANG="cs"
for MODEL in "${CS_MODELS[@]}"; do
  python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
    --lang ${LANG} --bert_model_name "${MODEL}"

  python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
    --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

  python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
done

LANG="de"
for MODEL in "${DE_MODELS[@]}"; do
  python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
    --lang ${LANG} --bert_model_name "${MODEL}"

  python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
    --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

  python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
done

LANG="en"
for MODEL in "${EN_MODELS[@]}"; do
  python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
    --lang ${LANG} --bert_model_name "${MODEL}"

  python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
    --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

  python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
done

LANG="zh"
for MODEL in "${ZH_MODELS[@]}"; do
  python3 get_emb_cluster_pca.py --data_cluster atomic --model bertsentence --cluster 2 --data context --dim 5 \
    --lang ${LANG} --bert_model_name "${MODEL}"

  python3 computeBERTAndGloVeScoreOfUserStudyActions.py --data_cluster atomic --cluster 2 --data context \
    --dim 5 --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"

  python3 plt_UserstudyCorr.py --lang ${LANG} --model_name "${MODEL}" --user_study "globalAMT"
done
