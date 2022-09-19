import json


def match_specification_file(sp_file, brick_list):
    new_sp_file = dict()
    for data, des in sp_file.items():
        words = des.split()
        max_share_num = 0
        max_share_index = 0
        min_brick_num = 999
        for i in range(len(brick_list)):
            shared_words = [word for word in words if word in brick_list[i]]
            shard_num = len(shared_words)
            brick_num = len(brick_list[i])
            if shard_num > max_share_num:
                max_share_num = shard_num
                max_share_index = i
                min_brick_num = brick_num
            if shard_num == max_share_num:
                if min_brick_num > brick_num:
                    max_share_num = shard_num
                    max_share_index = i
                    min_brick_num = brick_num
        if max_share_num > 0:
            new_sp_file[data] = brick_class_list[max_share_index]
    return new_sp_file


def generate_label(word_dict, sp_file):
    label_dict = dict()
    for id, data in word_dict.items():
        max_share_num = 0
        max_share_index = 0
        for key in sp_file.keys():
            words = key.split()
            share_words = [word for word in words if word in data]
            share_num = len(share_words)
            if share_num > max_share_num:
                max_share_num = share_num
                max_share_index = key
        label_dict[id] = sp_file[max_share_index]
    return label_dict


with open('../data/example_specification_file.json', 'r') as fp:
    specification_file = json.load(fp)

with open('../Brick/brick_class_list.json','r') as fp:
    brick_class_list = json.load(fp)

new_brick_list = list()
for i in range(len(brick_class_list)):
    new_brick_list.append(brick_class_list[i].split('_'))

new_sp_file = match_specification_file(specification_file, new_brick_list)
print(new_sp_file)

with open('../data/example_word_dict.json', 'r') as fp:
    word_dict = json.load(fp)

label_dict = generate_label(word_dict, new_sp_file)
print(label_dict)