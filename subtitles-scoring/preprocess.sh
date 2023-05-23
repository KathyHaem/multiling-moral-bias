#! /bin/bash

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/ar-cs.txt" --lang1 ar --lang2 cs --out_folder "../opensubtitles/ar-cs.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/ar-de.txt" --lang1 ar --lang2 de --out_folder "../opensubtitles/ar-de.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/ar-en.txt" --lang1 ar --lang2 en --out_folder "../opensubtitles/ar-en.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/ar-zh_cn.txt" --lang1 ar --lang2 zh_cn --out_folder "../opensubtitles/ar-zh_cn.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/cs-de.txt" --lang1 cs --lang2 de --out_folder "../opensubtitles/cs-de.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/cs-en.txt" --lang1 cs --lang2 en --out_folder "../opensubtitles/cs-en.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/cs-zh_cn.txt" --lang1 cs --lang2 zh_cn --out_folder "../opensubtitles/cs-zh_cn.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/de-en.txt" --lang1 de --lang2 en --out_folder "../opensubtitles/de-en.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/de-zh_cn.txt" --lang1 de --lang2 zh_cn --out_folder "../opensubtitles/de-zh_cn.tok"

python3 prepare_parallel_opus_files.py --in_folder "../opensubtitles/en-zh_cn.txt" --lang1 en --lang2 zh_cn --out_folder "../opensubtitles/en-zh_cn.tok"



