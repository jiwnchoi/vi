import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import dirichlet, poisson

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
        self.alpha = np.ones(n_topics)
        self.pi = dirichlet.rvs(self.alpha, size=self.n_docs)

        self.lamb = np.random.gamma(1, 1, size=(n_topics, self.n_vocabs))
        self.z = np.random.choice(n_topics, size=self.n_docs)

        self.log_likelihoods = []

    def fit(self, verbose: bool = False, early_stop: bool = True):
        self.log_likelihoods = []
        for iteration in range(self.n_iter):
            # (1) Update z
            sample_z = []
            for n in range(self.n_docs):
                znk = []  # 분모
                for k in range(self.n_topics):
                    var = poisson.pmf(self.matrix[n], self.lamb[k])
                    znk.append(np.log(self.pi[n, k]) + np.sum(np.log(var)))

                znk_n = (np.array(znk) - max(np.array(znk))) / 2  # overflow 방지
                zn_prop = np.exp(znk_n) / np.sum(
                    np.exp(znk_n)
                )  # 안정적으로 하기 위해서, softmax 로 변환

                sampling_zn = np.random.choice(
                    [i for i in range(self.n_topics)], size=1, p=zn_prop
                )
                sample_z.append(sampling_zn[0])
            self.z = np.array(sample_z)

            # (2) Update lambda -> 베타는 numpy 패키지 특이성 때문에, 역수로 넣어줘야 함. 그런데, derivation 에는 z 에 대한 dependency 없는데, 코드에는 잘 나온듯. 확인할것.
            for k in range(self.n_topics):
                mask = self.z == k  # Boolean mask for documents in topic k
                self.lamb[k] = np.random.gamma(
                    1 + np.sum(self.matrix[mask], axis=0),
                    1 / (1 + self.matrix[mask].shape[0]),
                )

            # (3) Update pi
            doc_lengths = np.sum(self.matrix, axis=1)
            # topic_counts = np.bincount(z, minlength=n_topics) ## 이것도 시험 해보기.
            topic_counts = np.bincount(
                self.z, weights=doc_lengths, minlength=self.n_topics
            )  ## 단어 개수가 많은 다큐먼트의 경우에는 토픽에 속하는 문서 개수에 가중치를 부여해준다.
            self.pi = dirichlet.rvs(1 + topic_counts, size=self.n_docs)

            # Compute log (joint)
            log_likelihood = np.sum(
                np.log(self.pi[np.arange(self.n_docs), self.z])
                + np.sum(self.matrix * np.log(self.lamb[self.z]), axis=1)
            )
            if verbose:
                print(f"Iteration {iteration}: {log_likelihood}")
            self.log_likelihoods.append(log_likelihood)

            # convergence 확인
            if early_stop and iteration > 1:
                if (
                    np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2])
                    < self.tolerance
                ):
                    print(f"Gibbs sampler converged after {iteration} iterations.")
                    break

    def show_logs(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.log_likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("Joint distribution")
        plt.title("Joint distribution by Iteration in Gibbs Sampling")
        plt.show()

    def show_topics(self, top_words: int = 10):
        for k in range(self.n_topics):
            top_word_indices = self.lamb[k, :].argsort()[-top_words:][::-1]
            top_words_k = self.vocabs[top_word_indices]
            print(f"Topic {k+1}: {', '.join(top_words_k)}")

    def show_topic(self, document_num: int):
        probable_topic = self.z[document_num] + 1
        document = self.docs[document_num]
        print(f"Document: {document}")
        print(f"Topic: {probable_topic}")
