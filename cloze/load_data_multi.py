import json
from keras_preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from utiliz import Dataset
import pickle

source_building_list = ['example']
target_building = 'example'

word_maxlen = 10
char_maxnum = 10

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
Load Relation Data
'''
source_word_dicts_list_relation = []
source_label_dicts_list_relation = []

for b in source_building_list:
    with open('../data/'+b+'_word_dict_relation.json', 'r') as fp:
        source_word_dicts_list_relation.append(json.load(fp))
with open('../data/'+target_building+'_word_dict_relation.json', 'r') as fp:
    target_word_dict_relation= json.load(fp)

for b in source_building_list:
    with open('../data/'+b+'_label_dict_relation.json', 'r') as fp:
        source_label_dicts_list_relation.append(json.load(fp))
with open('../data/'+target_building+'_label_dict_relation.json', 'r') as fp:
    target_label_dict_relation = json.load(fp)

X_word_source_relation = []
X_char_source_relation = []
Y_source_relation = []
X_word_target_relation = []
X_char_target_relation = []
Y_target_relation = []

for dict_index in range(len(source_word_dicts_list)):
    for key, value in source_word_dicts_list_relation[dict_index].items():
        chars_sub = []
        chars_obj = []
        for i in range(word_maxlen):
            if i < len(value['sub']):
                for j in range(char_maxnum):
                    if j < len(value['sub'][i]):
                        chars_sub.append(value['sub'][i][j])
                    else:
                        chars_sub.append('#')
            else:
                chars_sub.extend(['#']*char_maxnum)
        for i in range(word_maxlen):
            if i < len(value['obj']):
                for j in range(char_maxnum):
                    if j < len(value['obj'][i]):
                        chars_obj.append(value['obj'][i][j])
                    else:
                        chars_obj.append('#')
            else:
                chars_obj.extend(['#']*char_maxnum)
        X_char_source_relation.append([chars_sub, chars_obj])
        X_word_source_relation.append([source_word_dicts_list_relation[dict_index][key]['sub'], source_word_dicts_list_relation[dict_index][key]['obj']])
        Y_source_relation.append(source_label_dicts_list_relation[dict_index][key])

for key, value in target_word_dict_relation.items():
    chars_sub = []
    chars_obj = []
    for i in range(word_maxlen):
        if i < len(value['sub']):
            for j in range(char_maxnum):
                if j < len(value['sub'][i]):
                    chars_sub.append(value['sub'][i][j])
                else:
                    chars_sub.append('#')
        else:
            chars_sub.extend(['#'] * char_maxnum)
    for i in range(word_maxlen):
        if i < len(value['obj']):
            for j in range(char_maxnum):
                if j < len(value['obj'][i]):
                    chars_obj.append(value['obj'][i][j])
                else:
                    chars_obj.append('#')
        else:
            chars_obj.extend(['#'] * char_maxnum)
    X_char_target_relation.append([chars_sub, chars_obj])
    X_word_target_relation.append([target_word_dict_relation[key]['sub'], target_word_dict_relation[key]['obj']])
    Y_target_relation.append(target_label_dict_relation[key])

'''
build word and char vocab
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
for data in X_word_source_relation:
    for data_i in data:
        for chars in data_i:
            char_list.extend(chars)
        word_list.extend(data_i)
for data in X_word_target_relation:
    for data_i in data:
        for chars in data_i:
            char_list.extend(chars)
        word_list.extend(data_i)
char_list.append('#')
word_list = list(set(word_list))
char_list = list(set(char_list))

for i in range(len(word_list)):
    word_vocab[word_list[i]] = i+1
for i in range(len(char_list)):
    char_vocab[char_list[i]] = i+1


'''
Tokenize the words
'''

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

X_word_source = sequence.pad_sequences(X_word_source, padding='post', maxlen=word_maxlen).tolist()
X_word_target = sequence.pad_sequences(X_word_target, padding='post', maxlen=word_maxlen).tolist()

print(X_word_source)
print(X_word_target)
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
tokenize the relation data
'''


def tokenize_relation(raw_strings, vocab):
    for word_idx in range(len(raw_strings)):
        for idx in range(len(raw_strings[word_idx][0])):
            word = raw_strings[word_idx][0][idx]
            raw_strings[word_idx][0][idx] = vocab[word]
    for word_idx in range(len(raw_strings)):
        for idx in range(len(raw_strings[word_idx][1])):
            word = raw_strings[word_idx][1][idx]
            raw_strings[word_idx][1][idx] = vocab[word]
    return raw_strings

def pad_relation(word_relation, maxlen):
    for i in range(len(word_relation)):
        temp = sequence.pad_sequences(word_relation[i], padding='post', maxlen=maxlen).tolist()
        word_relation[i] = temp

X_word_source_relation = tokenize_relation(X_word_source_relation, word_vocab)
X_word_target_relation = tokenize_relation(X_word_target_relation, word_vocab)
X_char_source_relation = tokenize_relation(X_char_source_relation, char_vocab)
X_char_target_relation = tokenize_relation(X_char_target_relation, char_vocab)

pad_relation(X_word_source_relation, maxlen=word_maxlen)
pad_relation(X_word_target_relation, maxlen=word_maxlen)

'''
binarize the relation label
'''
label_list_relation = ['controls','haspoint','haspart','haslocation','hastag','ispointof','ispartof','islocationof','feeds','isfedby']
binarizer_relation = LabelBinarizer()
binarizer_relation.fit(label_list_relation)
Y_source_relation = binarizer_relation.transform(Y_source_relation)
Y_target_relation = binarizer_relation.transform(Y_target_relation)
'''
pack into Dataset
'''

dataset = Dataset(X_word_source, X_char_source, Y_source, X_word_target, X_char_target, Y_target)
with open('dataset/example.pkl', 'wb') as fp:
    pickle.dump(dataset, fp)

print(X_word_source_relation)
print(X_char_source_relation)
print(Y_source_relation)

print(X_word_target_relation)
print(X_char_target_relation)
dataset_relation = Dataset(X_word_source_relation, X_char_source_relation, Y_source_relation,X_word_target_relation, X_char_target_relation,Y_target_relation)
with open('dataset/example_relation.pkl', 'wb') as fp:
    pickle.dump(dataset_relation, fp)