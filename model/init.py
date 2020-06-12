from model.preprocess import Preprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_to_file1 = './data.txt'
path_to_file2 = './data2.txt'
path_to_file3 = './data3.txt'




a = Preprocess().read_text(path_to_file1)
b = Preprocess().read_text(path_to_file2)
c = Preprocess().read_text(path_to_file3)

a = Preprocess().stop_words_remove(a)
print(a)


print(Preprocess().jaro_similarity(a,c))
print(Preprocess().seq_matcher_similarity(a,c))


freq = Preprocess().count_words(b)
freq = dict(freq.most_common(7))
print(freq)



labels, values = zip(*freq.items())

indSort = np.argsort(values)[::-1]
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.01

plt.bar(indexes, values)
plt.xticks(indexes + bar_width, labels)
plt.show()

