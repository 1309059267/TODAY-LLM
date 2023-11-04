import json
from math import fabs
import os
import random
from types import new_class

root_dir = 'data/train'
name_list = os.listdir(root_dir)

# 构造医学任务数据集
all_lines = []
for name in name_list:
    if name[:-6] != 'MOSS' and name[:-6] != 'all_datas':
        file_path = os.path.join(root_dir, name)
        with open(file_path, encoding='utf8', mode='r') as f:
            lines = f.readlines() 
            all_lines += lines
data_num = len(all_lines)
sort_list = list(range(0, data_num))
random.shuffle(sort_list)
new_lines = []
for id in sort_list:
    new_lines.append(all_lines[id])

id = 1
with open('data/train/medical_datas.jsonl', 'w', encoding='utf8') as f:
    for line in new_lines:
        data = json.loads(line)
        data['conversation_id'] = id
        id += 1
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')



