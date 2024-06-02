import numpy as np
import re


class LDABase:
    def __init__(
        self,
        count_path: str,
        doc_path: str,
        vocab_path: str,
        n_topics: int,
        n_iter: int = 100,
        tolerance=1e-3,
    ) -> None:
        self.tolerance = tolerance
        self.n_topics = n_topics
        self.n_iter = n_iter
        with open(count_path, "r") as file:
            self.counts = file.readlines()

        self.matrix, self.filtered_indices = self._preprocess_documents(self.counts)

        with open(doc_path, "r") as file:
            docs = file.read()
            self.docs = re.findall(r"<TEXT>\n(.*?)\n </TEXT>", docs, re.DOTALL)

        with open(vocab_path, "r") as file:
            vocabs = [word.strip() for word in file.readlines()]
            self.vocabs = np.array(vocabs)[self.filtered_indices]

    def _preprocess_documents(self, documents: list[str]):
        doc_word_counts = []
        for document in documents:
            word_counts = document.split()[1:]
            word_counts = {
                int(wc.split(":")[0]): int(wc.split(":")[1]) for wc in word_counts
            }
            doc_word_counts.append(word_counts)

        vocab_len = max(max(d.keys()) for d in doc_word_counts if d) + 1
        input_matrix = np.zeros((len(doc_word_counts), vocab_len))
        for i, wc in enumerate(doc_word_counts):
            for word, count in wc.items():
                input_matrix[i, word] = count

        word_freqs = np.sum(input_matrix, axis=0)
        filtered_indices = np.where((word_freqs >= 10) & (word_freqs <= 1000))[0]
        return input_matrix[:, filtered_indices], filtered_indices

    def fit(self, verbose: bool = False, early_stop: bool = True):
        raise NotImplementedError

    def show_logs(self):
        raise NotImplementedError

    def show_topics(self, top_words: int):
        raise NotImplementedError

    def show_topic(self, document_num: int):
        raise NotImplementedError
