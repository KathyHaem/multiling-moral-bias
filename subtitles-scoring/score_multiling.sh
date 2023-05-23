
for LANG1 in cs de; do
  for LANG2 in en; do
    python3 score_tok_files.py \
      --tok_file "../opensubtitles/${LANG1}-${LANG2}.tok/OpenSubtitles.${LANG1}-${LANG2}.${LANG1}.tok" \
      --out_folder "../opensubtitles/${LANG1}-${LANG2}.tok/" --lang ${LANG1} \
      --model_base_folder "../MoralDirection/mort/results/bias/cluster" \
      --model_name "kathaem/xlm-roberta-base-sentence-transformer-nli-5langs"
    python3 score_tok_files.py \
      --tok_file "../opensubtitles/${LANG1}-${LANG2}.tok/OpenSubtitles.${LANG1}-${LANG2}.${LANG2}.tok" \
      --out_folder "../opensubtitles/${LANG1}-${LANG2}.tok/" --lang ${LANG2} \
      --model_base_folder "../MoralDirection/mort/results/bias/cluster" \
      --model_name "kathaem/xlm-roberta-base-sentence-transformer-nli-5langs"
  done
done