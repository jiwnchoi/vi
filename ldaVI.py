import numpy as np
from ldaBase import LDABase
from scipy.special import psi


class LDAVI(LDABase):
    def __init__(
        self,
        count_path: str,
        doc_path: str,
        vocab_path: str,
        n_topics: int,
        n_iter: int = 100,
        tolerance=1e-3,
    ) -> None:
        super().__init__(count_path, doc_path, vocab_path, n_topics, n_iter, tolerance)

        self.alpha = 0.1
        self.beta = 0.01
        self.n_docs, self.n_words = self.matrix.shape

        self.lamb = np.random.gamma(100.0, 1.0 / 100.0, (self.n_docs, self.n_topics))
        self.Elogbeta = psi(self.lamb) - (
            psi(np.sum(self.lamb, 1))[:, np.newaxis]
            if self.lamb.shape > 1
            else psi(np.sum(self.lamb))
        )
        self.expElogbeta = np.exp(self.Elogbeta)

        self.gamma = np.random.gamma(100.0, 1.0 / 100.0, (self.n_docs, self.n_topics))
