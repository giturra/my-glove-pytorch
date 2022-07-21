import torch
import re
from collections import Counter


from torch.utils.data import Dataset

class WordIndexer:
    """Transform g a dataset of text to a list of index of words. Not memory 
    optimized for big datasets"""

    def __init__(self, min_word_occurences=1, right_window=1, oov_word="OOV"):
        self.oov_word = oov_word
        self.right_window = right_window
        self.min_word_occurences = min_word_occurences
        self.word_to_index = {oov_word: 0}
        self.index_to_word = [oov_word]
        self.word_occurrences = {}
        self.re_words = re.compile(r"\b[a-zA-Z]{2,}\b")

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word_to_index[word]
        except KeyError:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word.append(word)
            return idx

    @property
    def n_words(self):
        return len(self.word_to_index)

    def fit_transform(self, texts):
        l_words = [list(self.re_words.findall(sentence.lower()))
                   for sentence in texts]
        word_occurrences = Counter(word for words in l_words for word in words)

        self.word_occurrences = {
            word: n_occurences
            for word, n_occurences in word_occurrences.items()
            if n_occurences >= self.min_word_occurences}

        oov_index = 0
        return [[self._get_or_set_word_to_index(word)
                 if word in self.word_occurrences else oov_index
                 for word in words]
                for words in l_words]

    def _get_ngrams(self, indexes):
        for i, left_index in enumerate(indexes):
            window = indexes[i + 1:i + self.right_window + 1]
            for distance, right_index in enumerate(window):
                yield left_index, right_index, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        z = 0
        for indexes in data:
            l_ngrams = self._get_ngrams(indexes)
            #print(l_ngrams)
            for left_index, right_index, distance in l_ngrams:
                comatrix[(left_index, right_index)] += 1. / distance
                z += 1
        return zip(*[(left, right, x) for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.n_obs

    def __init__(self, texts, right_window=1, random_state=0, min_count=1):
        torch.manual_seed(random_state)

        self.indexer = WordIndexer(min_word_occurences=min_count, right_window=right_window)
        data = self.indexer.fit_transform(texts)
        left, right, n_occurrences = self.indexer.get_comatrix(data)

        self.n_obs = len(left)

        self.left = torch.LongTensor(left)
        self.right = torch.LongTensor(right)
        self.n_occurences = torch.FloatTensor(n_occurrences)
    
    def __getitem__(self, idx):
        return self.left[idx], self.right[idx], self.n_occurences[idx]

