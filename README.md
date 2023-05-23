This is the repository for our paper [Speaking Multiple Languages Affects the Moral Bias of Language Models](https://arxiv.org/abs/2211.07733).
Its "MoralDirection" component is adapted from [Patrick Schramowski](https://github.com/ml-research/MoRT_NMI), and we only include the parts necessary for our paper in this repository.

# Installation

There is a `requirements.txt` file under `MoralDirection` which you can use to install dependencies.
This was also copied and pruned from the MoRT_NMI repository, so the versions are a little bit older now.
We used these versions with Python 3.8 to induce the moral dimension and score sentences.

There is another `requirements.txt` file under `subtitles-scoring`.
This is because the two were originally in two separate repositories.
You can probably (!) install these requirements in addition to the above ones if needed.


# Reproducing our Work

To induce the *moral dimension* for the models we mention in the paper, run:

```bash
cd MoralDirection
bash induce-md-plot-user-study.sh
```

For the subtitles scoring and score difference calculation:

1. Download [OpenSubtitles dataset](https://opus.nlpl.eu/OpenSubtitles-v2018.php) into a directory named `opensubtitles`.
2. Run these scripts:

```bash
cd subtitles-scoring
bash preprocess.sh  # comment out lines you don't need
bash score_cs-en.sh  # scores language pair with relevant monolingual models
bash score_de-en.sh
bash score_multiling.sh  # scores language pairs with multilingual model. analogous for other pairs
```

We applied additional filtering to the data based on translation fit using Marian as a translation model.
However, this was done on command line, so we can't share that script.
Check `notebooks/en-cs moral scores.ipynb` and `notebooks/en-de moral scores.ipynb` for analysis of the scored and filtered data.

To score the rephrased MFQ questions as in the paper, run:

```bash
cd MoralDirection
bash score-mfq-12.sh
```

Feel free to let us know if any of these don't work. 
Something may have got lost while cleaning up the repository for publishing.

The `notebooks` folder contains primarily code for generating the different graphs in the paper.

To train sentence-transformer models, check the [training examples](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/) 
in the sentence-transformers library. 


# Models Released

We release fine-tuned sentence-transformer versions of the models we used:

- [arabertv02 tuned on Arabic](https://huggingface.co/kathaem/aubmindlab-arabertv02-base-sentence-transformer-xnli-ar) MNLI data (released with XNLI)
- [RobeCzech tuned on Czech](https://huggingface.co/kathaem/ufal-robeczech-base-sentence-transformer-mnli-cs) MNLI data (machine translated by us)
- [bert-base-cased tuned on English](https://huggingface.co/kathaem/bert-base-cased-sentence-transformer-mnli-en) MNLI data (original)
- [gBert tuned on German](https://huggingface.co/kathaem/deepset-gbert-base-sentence-transformer-xnli-de) MNLI data (released with XNLI)
- [bert-base-chinese tuned on Chinese](https://huggingface.co/kathaem/bert-base-chinese-sentence-transformer-xnli-zh) MNLI data (released with XNLI)
- [XLM-R tuned on the five above datasets](https://huggingface.co/kathaem/xlm-roberta-base-sentence-transformer-nli-5langs)

Our Czech machine translated data can be found in `mnli-en-cs.txt.gz`.
For validation of each model, we used the 'dev' split from the STS dataset, 
again machine translated by us to the respective target languages.
See the paper for details.


# Citation

If you find our work useful, please cite our paper:

```
@misc{haemmerl2022speaking,
      title={Speaking Multiple Languages Affects the Moral Bias of Language Models}, 
      author={Katharina H\"ammerl and Bj\"orn Deiseroth and Patrick Schramowski and Jind\v{r}ich Libovick\'y and Constantin A. Rothkopf and Alexander Fraser and Kristian Kersting},
      year={2022},
      eprint={2211.07733},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
