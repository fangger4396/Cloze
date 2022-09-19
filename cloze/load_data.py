import json
from keras_preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
import numpy as np

source_building_list = ['example']
target_building = 'example'

'''
Load Entity Type Data
'''
source_word_dicts_list = []
source_label_dicts_list = []

for b in source_building_list:
    with open('../data/'+b+'_word_dict.json', 'r') as fp:
        source_word_dicts_list.append(json.load(fp))
with open('../data/'+target_building+'_word_dict.json', 'r') as fp:
    target_word_dict = json.load(fp)

for b in source_building_list:
    with open('../data/'+b+'_label_dict.json', 'r') as fp:
        source_label_dicts_list.append(json.load(fp))
with open('../data/'+target_building+'_label_dict.json', 'r') as fp:
    target_label_dict = json.load(fp)

X_word_source = []
X_char_source = []
Y_source = []
X_word_target = []
X_char_target = []
Y_target = []
word_maxlen = 10
char_maxnum = 10
for dict_index in range(len(source_word_dicts_list)):
    for key, value in source_word_dicts_list[dict_index].items():
        chars = []
        for i in range(word_maxlen):
            if i < len(value):
                for j in range(char_maxnum):
                    if j < len(value[i]):
                        chars.append(value[i][j])
                    else:
                        chars.append('#')
            else:
                chars.extend(['#']*char_maxnum)
        X_char_source.append(chars)
        X_word_source.append(source_word_dicts_list[dict_index][key])
        Y_source.append(source_label_dicts_list[dict_index][key])
print(X_char_source)
for key, value in target_word_dict.items():
    chars = []
    for i in range(word_maxlen):
        if i < len(value):
            for j in range(char_maxnum):
                if j < len(value[i]):
                    chars.append(value[i][j])
                else:
                    chars.append('#')
        else:
            chars.extend(['#'] * char_maxnum)
    X_char_target.append(chars)
    X_word_target.append(target_word_dict[key])
    Y_target.append(target_label_dict[key])

'''
Tokenize the words
'''
word_list = []
char_list = []
word_vocab = {}
char_vocab = {}
for data in X_word_source:
    for chars in data:
        char_list.extend(chars)
    word_list.extend(data)
for data in X_word_target:
    for chars in data:
        char_list.extend(chars)
    word_list.extend(data)
char_list.append('#')
word_list = list(set(word_list))
char_list = list(set(char_list))

for i in range(len(word_list)):
    word_vocab[word_list[i]] = i+1
for i in range(len(char_list)):
    char_vocab[char_list[i]] = i+1


def tokenize(raw_strings, vocab):
    for word_idx in range(len(raw_strings)):
        for idx in range(len(raw_strings[word_idx])):
            word = raw_strings[word_idx][idx]
            raw_strings[word_idx][idx] = vocab[word]
    return raw_strings


X_word_source = tokenize(X_word_source, word_vocab)
X_word_target = tokenize(X_word_target, word_vocab)
X_char_source = tokenize(X_char_source, char_vocab)
X_char_target = tokenize(X_char_target, char_vocab)

X_word_source = sequence.pad_sequences(X_word_source, padding='post', maxlen=10)
X_word_target = sequence.pad_sequences(X_word_target, padding='post', maxlen=10)

print(X_word_source)
print(X_char_source)

'''
tokenize the labels
'''


binarizer = LabelBinarizer()
with open('../Brick/brick_class_list.json', 'r') as fp:
    brick_class_list = json.load(fp)
label_dict = {}
cnt = 0
for cls in brick_class_list:
    label_dict[cls] = cnt
    cnt += 1

binarizer.fit(brick_class_list)
Y_source = binarizer.transform(Y_source)
Y_target = binarizer.transform(Y_target)

'''
pack into Dataset
'''


class Dataset():
    def __init__(self, X_word_source, X_char_source, Y_source, X_word_target, X_char_target, Y_target):
        self.X_source = [np.array(X_word_source), np.array(X_char_source)]
        self.Y_source = np.array(Y_source)
        self.X_target = [np.array(X_word_target), np.array(X_char_target)]
        self.Y_target = np.array(Y_target)


dataset = Dataset(X_word_source, X_char_source, Y_source, X_word_target, X_char_target, Y_target)
import pickle
with open('dataset/example.pkl', 'wb') as fp:
    pickle.dump(dataset, fp)