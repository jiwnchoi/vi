import numpy as np
from matplotlib import pyplot as plt
from scipy.special import psi, gammaln
from tqdm import tqdm

from ldaBase import LDABase


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
        sums = np.sum(self.matrix, axis=1)
        self.matrix = self.matrix[sums > 0]

        self.X = self.matrix / np.sum(self.matrix, axis=1, keepdims=True) * 10
        self.N, self.V = self.X.shape
        self.K = n_topics

        self.phi = np.random.dirichlet(np.ones(self.K), size=self.N)
        self.gamma = np.random.gamma(100.0, 0.01, (self.K))
        self.alpha = np.abs(np.random.normal(0.1, 0.001, (self.K, self.V)))
        self.beta = np.abs(np.random.normal(0.1, 0.001, (self.K, self.V)))
        self.gamma_lnx = gammaln(self.X + 1)
        for _ in range(5):
            self._m_step()

    def _e_step(self):
        self.E_log_lambda = psi(self.alpha) - np.log(self.beta)  # (K, V)
        self.E_log_pi = Edirichlet(self.gamma)  # (K,)
        self.E_lambda = self.alpha / self.beta  # (K,V)

        log_phi = (
            np.dot(self.X, self.E_log_lambda.T)  # (N, K)
            - np.sum(self.E_lambda, axis=1)  # (K,)
            + self.E_log_pi  # (K,)
            - np.sum(self.gamma_lnx, axis=1, keepdims=True)  # (N,)
        )

        logsumexp = np.max(log_phi, axis=1, keepdims=True) + np.log(
            np.sum(
                np.exp(log_phi - np.max(log_phi, axis=1, keepdims=True)),
                axis=1,
                keepdims=True,
            )
        )

        self.phi = np.exp(log_phi - logsumexp)  # (N, K)

    def _m_step(self):
        A = np.dot(self.phi.T, self.X)
        B = np.sum(self.phi, axis=0)

        self.alpha = self.alpha + A
        self.beta = self.beta + B[:, np.newaxis]
        self.gamma = self.gamma + B

    def _compute_elbo(self):
        E_log_p_x = np.sum(
            np.multiply(self.phi, np.matmul(self.X, self.E_log_lambda.T))
        ) - np.sum(np.matmul(self.phi, self.E_lambda))

        E_log_p_z = np.sum(np.matmul(self.phi, self.E_log_pi))
        E_log_q_z = np.sum(np.multiply(self.phi, np.log(self.phi + 1e-100)))
        return E_log_p_x + E_log_p_z - E_log_q_z

    def fit(self, verbose: bool = False, early_stop: bool = True):
        p_bar = tqdm(range(self.n_iter))
        self.elbos = []

        for i in p_bar:
            self._e_step()
            for i in range(5):
                self._m_step()
            bound = self._compute_elbo()
            self.elbos.append(bound)

            if (
                early_stop
                and i > 0
                and np.abs(self.elbos[-1] - self.elbos[-2]) < self.tolerance
            ):
                break

            p_bar.set_postfix({"ELBO": bound})
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
            top_word_indices = np.argsort(self.E_lambda[k])[-top_words:][::-1]
            print(f"Topic {k+1}: {', '.join(self.vocabs[top_word_indices])}")

    def show_doc_topic(self, document_num: int):
        probable_topic = np.argmax(self.phi[document_num]) + 1
        return probable_topic

    def infer_topics(self, doc_term_matrix: np.ndarray):
        for i in range(self.n_iter):
            E_log_lambda = psi(self.alpha) - np.log(self.beta)
            E_log_pi = Edirichlet(self.gamma)

            log_phi = (
                np.dot(doc_term_matrix, E_log_lambda.T)
                - np.sum(self.alpha / self.beta, axis=1)
                + E_log_pi
                - np.sum(gammaln(doc_term_matrix + 1), axis=1, keepdims=True)
            )

            logsumexp = np.max(log_phi, axis=1, keepdims=True) + np.log(
                np.sum(
                    np.exp(log_phi - np.max(log_phi, axis=1, keepdims=True)),
                    axis=1,
                    keepdims=True,
                )
            )

            phi = np.exp(log_phi - logsumexp)
        return np.argmax(phi, axis=1) + 1

    def export_model(self, path: str):
        np.savez(
            path,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            phi=self.phi,
            elbos=self.elbos,
        )

    def load_model(self, path: str):
        data = np.load(path)
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.gamma = data["gamma"]
        self.phi = data["phi"]
        self.elbos = data["elbos"]
        self.E_lambda = self.alpha / self.beta
        self.E_log_lambda = psi(self.alpha) - np.log(self.beta)
        self.E_log_pi = Edirichlet(self.gamma)
        self.gamma_lnx = gammaln(self.X + 1)
