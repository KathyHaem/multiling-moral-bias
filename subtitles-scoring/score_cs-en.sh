python3 score_tok_files.py \
      --tok_file "../opensubtitles/cs-en.tok/OpenSubtitles.cs-en.en.tok" \
      --out_folder "../opensubtitles/cs-en.tok/" --lang en \
      --model_base_folder "../MoralDirection/mort/results/bias/cluster" \
      --model_name "kathaem/bert-base-cased-sentence-transformer-mnli-en"

python3 score_tok_files.py \
      --tok_file "../opensubtitles/cs-en.tok/OpenSubtitles.cs-en.cs.tok" \
      --out_folder "../opensubtitles/cs-en.tok/" --lang cs \
      --model_base_folder "../MoralDirection/mort/results/bias/cluster" \
      --model_name "kathaem/ufal-robeczech-base-sentence-transformer-mnli-cs"
