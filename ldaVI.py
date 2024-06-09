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
        self.alpha = 0.1
        self.beta = 0.01
        self.n_docs, self.n_words = self.matrix.shape

        self.lamb = np.random.gamma(100, 0.01, (self.n_topics, self.n_words))
        self.Elogbeta = Edirichlet(self.lamb)
        self.expElogbeta = np.exp(self.Elogbeta)

        self.gamma = np.random.gamma(100, 0.01, (self.n_docs, self.n_topics))
        self.n_iter_gamma = 5

    def _e_step(self):
        sstats = np.zeros((self.n_topics, self.n_words))
        expElogtheta = np.exp(Edirichlet(self.gamma))

        for d in range(self.n_docs):
            count = self.matrix[d, :]
            expElogtheta_d = expElogtheta[d, :]
            phinorm = np.dot(expElogtheta_d, self.expElogbeta) + 1e-100

            for _ in range(self.n_iter_gamma):
                last_gamma = self.gamma[d, :].copy()
                self.gamma[d, :] = self.alpha + expElogtheta_d * np.dot(
                    count / phinorm, self.expElogbeta.T
                )

                expElogtheta_d = np.exp(Edirichlet(self.gamma[d, :]))
                phinorm = np.dot(expElogtheta_d, self.expElogbeta) + 1e-100

                if np.mean(np.abs(self.gamma[d, :] - last_gamma)) < self.tolerance:
                    break
            # print(sstats.shape, np.outer(expElogtheta_d, count / phinorm).shape)
            sstats += np.outer(expElogtheta_d, count / phinorm)

        sstats *= self.expElogbeta

        return sstats

    def _m_step(self, sstats):
        self.lamb = self.beta + sstats
        self.Elogbeta = Edirichlet(self.lamb)
        self.expElogbeta = np.exp(self.Elogbeta)

    def _compute_bound(self):
        Elogtheta = Edirichlet(self.gamma)
        expElogtheta = np.exp(Elogtheta)

        bound = np.sum(self.matrix * np.log(np.dot(expElogtheta, self.expElogbeta)))

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
        p_bar = tqdm(range(self.n_iter))
        self.elbos = []

        for i in p_bar:
            sstats = self._e_step()
            self._m_step(sstats)

            if i % 1 == 0:
                bound = self._compute_bound()
                self.elbos.append(bound)
                p_bar.set_postfix({"ELBO": bound})

            if (
                early_stop
                and i > 0
                and len(self.elbos) > 2
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

    def infer_topics(self, doc_term_matrix: np.ndarray):
        """
        Infer topic distributions for new documents using the trained LDA model.

        Parameters:
        doc_term_matrix (np.ndarray): Document-term matrix for new documents.

        Returns:
        np.ndarray: Inferred topic distributions for each document.
        """
        n_new_docs = doc_term_matrix.shape[0]
        gamma_new = np.random.gamma(100, 0.01, (n_new_docs, self.n_topics))
        expElogtheta_new = np.exp(Edirichlet(gamma_new))

        for d in range(n_new_docs):
            count = doc_term_matrix[d, :]
            expElogtheta_d = expElogtheta_new[d, :]
            phinorm = np.dot(expElogtheta_d, self.expElogbeta) + 1e-100

            for _ in range(self.n_iter_gamma):
                last_gamma = gamma_new[d, :].copy()
                gamma_new[d, :] = self.alpha + expElogtheta_d * np.dot(
                    count / phinorm, self.expElogbeta.T
                )

                expElogtheta_d = np.exp(Edirichlet(gamma_new[d, :]))
                phinorm = np.dot(expElogtheta_d, self.expElogbeta) + 1e-100

                if np.mean(np.abs(gamma_new[d, :] - last_gamma)) < self.tolerance:
                    break

        return np.argmax(gamma_new, axis=1) + 1

    def export_model(self, path: str):
        np.savez_compressed(path, gamma=self.gamma, lamb=self.lamb, elbos=self.elbos)

    def load_model(self, path: str):
        npzfile = np.load(path)
        self.gamma = npzfile["gamma"]
        self.lamb = npzfile["lamb"]
        self.elbos = npzfile["elbos"]
        self.Elogbeta = Edirichlet(self.lamb)
        self.expElogbeta = np.exp(self.Elogbeta)
