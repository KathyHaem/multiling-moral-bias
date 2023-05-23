import pickle
from typing import Union, List

import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer

""" Reproduced from MoralDirection/MoRT/funcs_mcm because the subtitles scoring was originally in a separate repository.
Adjusted to our needs/to be minimum working.
"""


class BERTSentenceSubspace:

    def __init__(self, filename_pickled_cluster, device="cpu",
                 transformer_model='bert-large-nli-mean-tokens', norm=None):

        self.device = device
        self.transformer_model = transformer_model
        sbert = SentenceTransformer(self.transformer_model, device=self.device)
        self.sbert = sbert
        self.sbert.eval()

        moralprojection_model = pickle.load(open(filename_pickled_cluster, "rb"))
        self.pca = moralprojection_model["projection"]
        if norm is None:
            Y = moralprojection_model["Y"]
            Y[:, 0] = -Y[:, 0]
            norm = max(np.absolute(Y[:, 0]))
        self.norm = norm

        if "sign" not in list(moralprojection_model.keys()):
            self.moral_score_sign = -1
        else:
            self.moral_score_sign = moralprojection_model["sign"]

    def numpy_pca_transform(self, emb, norm=None):
        if len(emb.shape) == 1:
            emb = [emb]
        X_transformed = self.pca.transform(emb)
        if norm is not None:
            X_transformed /= norm
        return X_transformed

    def bias(self, statement, is_batch=True, norm=None, pre_tokenised=False, batch_size=32) -> List[float]:
        if not statement:
            return []
        if norm is None:
            norm = self.norm

        mcm_input = statement if is_batch else [statement]
        embs = self.get_sen_embedding(mcm_input, pre_tokenised, show_progress_bar=True, batch_size=batch_size)

        pca_embs = self.numpy_pca_transform(embs, norm)
        scores = [pca[0] for pca in pca_embs]
        return self.moral_score_sign * np.array(scores)

    def get_sen_embedding(self, messages, pre_tokenised=False, dtype='numpy', show_progress_bar=False, batch_size=64
                          ) -> Union[ndarray, List]:
        if not messages:
            return []
        sentence_embeddings = self.sbert.encode(
            messages,
            device=self.device,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            is_split_into_words=pre_tokenised
        )
        if dtype == 'numpy':  # this is the default anyway, and encode defaults to outputting ndarray apparently
            return np.array(sentence_embeddings)
        elif dtype == 'list':
            return np.array(sentence_embeddings).tolist()
        else:
            raise ValueError("resulting dtype unknown")

