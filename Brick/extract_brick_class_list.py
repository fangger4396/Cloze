
from load_brick import pointTagsetList        as  point_tagsets, \
    locationTagsetList     as  location_tagsets, \
    equipTagsetList        as  equip_tagsets, \
    tagsetTree as  tagset_tree


def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list] \
                     + added_d_list
    return d_list


tagset_list = equip_tagsets + location_tagsets + point_tagsets
new_tagset_list = tree_flatter(tagset_tree, [])
print(new_tagset_list)

import json
with open('brick_class_list.json', 'w') as fp:
    json.dump(new_tagset_list, fp, indent=2)

