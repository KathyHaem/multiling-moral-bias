#!/bin/bash

set -x

for LANG in ar de cs en zh; do
  python3 scoreMFQ.py --lang ${LANG} --model_name "kathaem/xlm-roberta-base-sentence-transformer-nli-5langs" --mfq 12
  python3 scoreMFQ.py --lang ${LANG} --model_name "xlm-roberta-base" --mfq 12
  python3 scoreMFQ.py --lang ${LANG} --model_name "bert-base-multilingual-cased" --mfq 12
done

python3 scoreMFQ.py --lang ar --model_name "output/arabertv02_nliv2_ar" --mfq 12
python3 scoreMFQ.py --lang ar --model_name "aubmindlab/bert-base-arabertv02" --mfq 12

python3 scoreMFQ.py --lang de --model_name "output/gbert-base_nliv2_de" --mfq 12
python3 scoreMFQ.py --lang de --model_name "deepset/gbert-base" --mfq 12

python3 scoreMFQ.py --lang cs --model_name "output/robeczech-base_nliv2_cs" --mfq 12
python3 scoreMFQ.py --lang cs --model_name "ufal/robeczech-base" --mfq 12

python3 scoreMFQ.py --lang en --model_name "output/bert-base-cased_nliv2_en" --mfq 12
python3 scoreMFQ.py --lang en --model_name "bert-base-cased" --mfq 12

python3 scoreMFQ.py --lang zh --model_name "output/bert-base-chinese_nliv2_zh" --mfq 12
python3 scoreMFQ.py --lang zh --model_name "bert-base-chinese" --mfq 12
