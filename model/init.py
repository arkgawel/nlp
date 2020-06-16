from model.preprocess import Preprocess


path_to_file1 = './data.txt'
path_to_file2 = './data2.txt'
path_to_file3 = './data3.txt'


a = Preprocess().read_text(path_to_file1)
b = Preprocess().read_text(path_to_file2)
c = Preprocess().read_text(path_to_file3)

a = Preprocess().stop_words_remove(a)
b = Preprocess().stop_words_remove(b)
c = Preprocess().stop_words_remove(c)



print(Preprocess().jaro_similarity(a,c))
print(Preprocess().seq_matcher_similarity(a,c))


freq = Preprocess().count_words(c)

print(c)

Preprocess().draw_histogram(freq, 5)


