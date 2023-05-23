python3 score_tok_files.py \
      --tok_file "../opensubtitles/de-en.tok/OpenSubtitles.de-en.de.tok" \
      --out_folder "../opensubtitles/de-en.tok/" --lang de \
      --model_base_folder "../MoRT_NMI/MoRT/mort/results/bias/cluster" \
      --model_name "kathaem/deepset-gbert-base-sentence-transformer-xnli-de"

python3 score_tok_files.py \
      --tok_file "../opensubtitles/de-en.tok/OpenSubtitles.de-en.en.tok" \
      --out_folder "../opensubtitles/de-en.tok/" --lang en \
      --model_base_folder "../MoRT_NMI/MoRT/mort/results/bias/cluster" \
      --model_name "kathaem/bert-base-cased-sentence-transformer-mnli-en"