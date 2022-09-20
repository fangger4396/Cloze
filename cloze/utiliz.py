import json

word_dict_list = []
with open('../data/example_word_dict.json', 'r') as fp:
    word_dict = json.load(fp)
word_dict_list.append(word_dict)
word_dict_list.append(word_dict)


def extract_word_vocab(word_dict_list):
    word_vocab = []
    for word_dict in word_dict_list:
        for key, value in word_dict.items():
            word_vocab.extend(value)
    word_vocab = list(set(word_vocab))
    return word_vocab


def tf_detect(word_vocab, word_dict_list):
    word_tf_dict = {}
    for word in word_vocab:
        word_tf_dict[word] = 0
    for i in range(len(word_dict_list)):
        for key, value in word_dict_list[i].items():
            for word in value:
                word_tf_dict[word] += 1
    return word_tf_dict


def tf_idf_detect(word_vocab, word_dict_list):
    import math
    word_tf_dict = {}
    for word in word_vocab:
        word_tf_dict[word] = [0] * len(word_dict_list)
    for i in range(len(word_dict_list)):
        for key, value in word_dict_list[i].items():
            for word in value:
                word_tf_dict[word][i] += 1
    for word in word_tf_dict.keys():
        tf_idf = []
        for idx in range(len(word_tf_dict[word])):
            cnt = word_tf_dict[word][idx]
            tf_idf.append(
                cnt * math.log2((len(word_dict_list) + 1) / sum([1 if v != 0 else 0 for v in word_tf_dict[word]])))
        word_tf_dict[word] = tf_idf

    return word_tf_dict


def coreference_detect(word_vocab):
    import Levenshtein
    word_sim_dict = {}
    for word in word_vocab:
        sim_dict = {}
        for word2 in word_vocab:
            if word != word2:
                sim_dict[word2] = Levenshtein.distance(word2, word)
        word_sim_dict[word] = sim_dict
    return word_sim_dict


def coreference_generate(word_sim_dict, metadata):
    from numpy.random import choice
    metadata_gen = []
    for word in metadata:
        if word in word_sim_dict.keys():
            pa = []
            a = []
            for key, value in word_sim_dict[word].items():
                pa.append(value)
                a.append(key)
            pa_sum = sum(pa)
            for i in range(len(pa)):
                pa[i] = pa[i] / pa_sum
            word_ = choice(a, size=1, p=pa)[0]
            metadata_gen.append(word_)
        else:
            metadata_gen.append(word)
    return metadata_gen
