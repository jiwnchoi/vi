from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from ldaBase import LDABase
from scipy.special import psi, gammaln


def Edirichlet(alpha: np.ndarray) -> np.ndarray:
    return psi(alpha) - (
        psi(np.sum(alpha, axis=1))[:, np.newaxis]
        if alpha.ndim > 1
        else psi(np.sum(alpha))
    )


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
        self.alpha = np.full(n_topics, 0.1)
        self.beta = 0.01
        self.n_docs, self.n_words = self.matrix.shape

        self.lamb = np.random.gamma(100.0, 1.0 / 100.0, (self.n_topics, self.n_words))
        self.Elogbeta = Edirichlet(self.lamb)
        self.expElogbeta = np.exp(self.Elogbeta)

        self.gamma = np.random.gamma(100.0, 1.0 / 100.0, (self.n_docs, self.n_topics))

    def _e_step(self):
        sstats = np.zeros((self.n_topics, self.n_words))

        for d in range(self.n_docs):
            gamma_d = self.gamma[d, :]
            count = self.matrix[d, :]
            Elogtheta_d = Edirichlet(gamma_d)

            for _ in range(self.n_iter):
                last_gamma = gamma_d.copy()
                phi = np.exp(Elogtheta_d[:, np.newaxis] + self.Elogbeta)
                phi /= np.sum(phi, axis=0, keepdims=True)
                gamma_d = self.alpha + np.dot(count, phi.T)
                Elogtheta_d = Edirichlet(gamma_d)

                if np.mean((gamma_d - last_gamma) ** 2) < self.tolerance:
                    break

            self.gamma[d, :] = gamma_d
            sstats += np.dot(gamma_d[:, np.newaxis], count[np.newaxis, :])

        return sstats

    def _m_step(self, sstats):
        self.lamb = self.beta + sstats
        self.Elogbeta = Edirichlet(self.lamb)
        self.expElogbeta = np.exp(self.Elogbeta)

    def _compute_bound(self):
        bound = 0
        Elogtheta = Edirichlet(self.gamma)

        for d in range(self.n_docs):
            count = self.matrix[d, :]
            expElogtheta_d = np.exp(Elogtheta[d, :])
            bound += np.sum(count * np.log(np.dot(expElogtheta_d, self.expElogbeta)))

        alpha_term = gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))
        alpha_term += np.sum((self.alpha - 1) * Elogtheta)
        gamma_term = gammaln(np.sum(self.gamma, axis=1)) - np.sum(
            gammaln(self.gamma), axis=1
        )
        gamma_term -= np.sum((self.gamma - 1) * Elogtheta, axis=1)
        bound += np.sum(alpha_term - gamma_term)

        eta = self.beta
        lamb_term = gammaln(np.sum(eta)) - np.sum(gammaln(eta))
        lamb_term += np.sum((eta - 1) * self.Elogbeta)
        lambda_term = gammaln(np.sum(self.lamb, axis=1)) - np.sum(
            gammaln(self.lamb), axis=1
        )
        lambda_term -= np.sum((self.lamb - 1) * self.Elogbeta, axis=1)
        bound += lamb_term - np.sum(lambda_term)

        return bound

    def fit(self, verbose: bool = False, early_stop: bool = True):
        self.elbos = []

        for i in tqdm(range(self.n_iter)):
            sstats = self._e_step()
            self._m_step(sstats)
            bound = self._compute_bound()
            self.elbos.append(bound)

            if (
                early_stop
                and i > 0
                and np.abs(self.elbos[-1] - self.elbos[-2]) < self.tolerance
            ):
                break

            if verbose:
                print(f"Iteration {i+1}/{self.n_iter}: ELBO = {bound}")

    def plot_logs(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.elbos)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.title("ELBO by Iteration in Variational Inference")
        plt.show()

    def show_topics(self, top_words: int):
        for k in range(self.n_topics):
            top_word_indices = np.argsort(self.lamb[k])[-top_words:][::-1]
            print(f"Topic {k+1}: {', '.join(self.vocabs[top_word_indices])}")

    def show_doc_topic(self, document_num: int):
        probable_topic = np.argmax(self.gamma[document_num]) + 1
        document = self.docs[document_num]
        print(f"Document: {document}")
        print(f"Topic: {probable_topic}")
