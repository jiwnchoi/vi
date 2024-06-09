from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet
from scipy.special import logsumexp


from ldaBase import LDABase


class LDAGibbs(LDABase):
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
        self.n_docs, self.n_vocabs = self.matrix.shape
        self.alpha = np.full(n_topics, 0.1)
        self.pi = dirichlet.rvs(self.alpha, size=self.n_docs)

        self.lamb = np.random.gamma(100, 0.01, size=(n_topics, self.n_vocabs))
        self.z = np.random.choice(n_topics, size=self.n_docs)

        self.log_likelihoods = []

    def fit(self, verbose: bool = False, early_stop: bool = True):
        p_bar = tqdm(range(self.n_iter))
        num_docs, vocab_size = self.matrix.shape
        self.log_likelihoods = []

        for iteration in p_bar:
            # Update topic assignments
            log_word_distributions = np.log(self.lamb)
            log_topic_prob = np.log(self.pi) + np.einsum(
                "ij,kj->ik", self.matrix, log_word_distributions
            )
            log_topic_prob = log_topic_prob - logsumexp(
                log_topic_prob, axis=1, keepdims=True
            )
            topic_prob = np.exp(log_topic_prob)
            self.z = np.array(
                [
                    np.random.choice(self.n_topics, p=topic_prob_doc)
                    for topic_prob_doc in topic_prob
                ]
            )

            # Update word distributions
            for k in range(self.n_topics):
                mask = self.z == k
                self.lamb[k] = np.random.gamma(
                    1 + np.sum(self.matrix[mask], axis=0), 1 / (1 + mask.sum())
                )

            # Update topic proportions
            doc_lengths = np.sum(self.matrix, axis=1)
            topic_counts = np.bincount(
                self.z, weights=doc_lengths, minlength=self.n_topics
            )
            self.pi = dirichlet.rvs(1 + topic_counts, size=num_docs)

            # Compute log likelihood
            log_likelihood = np.sum(
                np.log(self.pi[np.arange(num_docs), self.z])
                + np.sum(
                    self.matrix * np.log(self.lamb[self.z]),
                    axis=1,
                )
            )
            self.log_likelihoods.append(log_likelihood)
            p_bar.set_postfix({"Log likelihood": log_likelihood})

            # Check for convergence
            if (
                early_stop
                and iteration > 1
                and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2])
                < self.tolerance
            ):
                print(f"Gibbs sampler converged after {iteration} iterations.")
                break

    def plot_logs(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.log_likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("Joint distribution")
        plt.title("Joint distribution by Iteration in Gibbs Sampling")
        plt.show()

    def export_model(self, path: str):
        np.savez_compressed(
            path,
            lamb=self.lamb,
            pi=self.pi,
            z=self.z,
            log_likelihoods=self.log_likelihoods,
        )

    def load_model(self, path: str):
        npzfile = np.load(path)
        self.lamb = npzfile["lamb"]
        self.pi = npzfile["pi"]
        self.z = npzfile["z"]
        self.log_likelihoods = npzfile["log_likelihoods"]

    def show_topics(self, top_words: int = 10):
        for k in range(self.n_topics):
            top_word_indices = self.lamb[k, :].argsort()[-top_words:][::-1]
            top_words_k = self.vocabs[top_word_indices]
            print(f"Topic {k+1}: {', '.join(top_words_k)}")

    def show_doc_topic(self, document_num: int):
        probable_topic = self.z[document_num] + 1
        document = self.docs[document_num]
        print(f"Document: {document}")
        print(f"Topic: {probable_topic}")

    def inter_topics(self, doc_term_matrix: np.ndarray):
        # doc_term_matrix: (n_docs, n_vocabs), new unseen documents
        num_docs, vocab_size = doc_term_matrix.shape
        pi_new = dirichlet.rvs(self.alpha, size=num_docs)
        z_new = np.random.choice(self.n_topics, size=num_docs)

        for iteration in range(self.n_iter):
            # Update topic assignments
            log_word_distributions = np.log(self.lamb)
            log_topic_prob = np.log(pi_new) + np.einsum(
                "ij,kj->ik", doc_term_matrix, log_word_distributions
            )
            log_topic_prob = log_topic_prob - logsumexp(
                log_topic_prob, axis=1, keepdims=True
            )
            topic_prob = np.exp(log_topic_prob)
            z_new = np.array(
                [
                    np.random.choice(self.n_topics, p=topic_prob_doc)
                    for topic_prob_doc in topic_prob
                ]
            )

            # Update topic proportions
            doc_lengths = np.sum(doc_term_matrix, axis=1)
            topic_counts = np.bincount(
                z_new, weights=doc_lengths, minlength=self.n_topics
            )
            pi_new = dirichlet.rvs(1 + topic_counts, size=num_docs)

        return z_new + 1
