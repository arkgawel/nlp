from difflib import SequenceMatcher
from collections import Counter
import jellyfish
import re
import numpy as np
import matplotlib.pyplot as plt

stop_words = ['w','ach', 'aj', 'albo', 'bardzo', 'bez', 'bo', 'być', 'ci', 'cię', 'ciebie', 'co', 'czy', 'daleko', 'dla', 'dlaczego', 'dlatego', 'do', 'dobrze', 'dokąd', 'dość', 'dużo', 'dwa', 'dwaj', 'dwie', 'dwoje', 'dziś', 'dzisiaj', 'gdyby', 'gdzie', 'go', 'ich', 'ile', 'im', 'inny', 'ja', 'ją', 'jak', 'jakby', 'jaki', 'je', 'jeden', 'jedna', 'jedno', 'jego', 'jej', 'jemu', 'jeśli', 'jest', 'jestem', 'jeżeli', 'już', 'każdy', 'kiedy', 'kierunku', 'kto', 'ku', 'lub', 'ma', 'mają', 'mam', 'mi', 'mną', 'mnie', 'moi', 'mój', 'moja', 'moje', 'może', 'mu', 'my', 'na', 'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze', 'natychmiast', 'nią', 'nic', 'nich', 'nie', 'niego', 'niej', 'niemu', 'nigdy', 'nim', 'nimi', 'niż', 'obok', 'od', 'około', 'on', 'ona', 'one', 'oni', 'ono', 'owszem', 'po', 'pod', 'ponieważ', 'przed', 'przedtem', 'są', 'sam', 'sama', 'się', 'skąd', 'tak', 'taki', 'tam', 'ten', 'to', 'tobą', 'tobie', 'tu', 'tutaj', 'twoi', 'twój', 'twoja', 'twoje', 'ty', 'wam', 'wami', 'was', 'wasi', 'wasz', 'wasza', 'wasze', 'we', 'więc', 'wszystko', 'wtedy', 'wy', 'żaden', 'zawsze', 'że']

class Preprocess:
    def read_text(self, path_to_file):
        with open(path_to_file, "r", encoding ="utf8") as file:
            text = file.read()
            text = text.replace("\n", "").replace("\r", "")
            text = text.lower()
        return text

    def stop_words_remove(self, text):
        result = [word for word in re.split("\W+",text) if word.lower() not in stop_words]
        space = ' '
        result = space.join(result)
        return result

    def count_words(self, text):
        text = text.lower()
        skips = [".", ", ", ":", ";", "'", '"']
        for ch in skips:
            text = text.replace(ch, "")
        word_counts = Counter(text.split(" "))
        return word_counts

    def jaro_similarity(self, a, b):
        result = jellyfish.jaro_similarity(a,b)
        return result

    def seq_matcher_similarity (self, a, b):
        return SequenceMatcher(None, a,  b).ratio()


    def draw_histogram(self, freq, n_elements):
        self.freq = dict(freq.most_common(n_elements))
        labels, values = zip(*self.freq.items())
        sorted_indicator = np.argsort(values)[::-1]
        labels = np.array(labels)[sorted_indicator]
        values = np.array(values)[sorted_indicator]
        prepared = np.arange(len(labels))

        plt.bar(prepared, values)
        plt.xticks(prepared, labels)
        plt.show()

