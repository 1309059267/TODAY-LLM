from cgi import print_arguments
import json
from math import fabs
import os
import random
import time
from types import new_class
from tqdm import tqdm

# with open('data/train/RE.jsonl', 'r', encoding='utf8') as f:
#     lines = f.readlines()

# with open('data/train/new_RE.jsonl', 'w', encoding='utf8') as f:
#     for line in lines:
#         data = json.loads(line)
#         new_data = {}
#         new_data['conversation_id'] = data['conversation_id']
#         new_data['category'] = data['category']
#         conversation = data['conversation']
#         new_conversation = {}
#         new_conversation['human'] = conversation[0]['human']
#         new_conversation['assistant'] = conversation[0]['answer']
#         new_data['conversation'] = [new_conversation]
#         json.dump(new_data, f, ensure_ascii=False)
#         f.write('\n')
# a = {
#     "姓名": ['a', 'b', 'c'],
#     "年龄": ['19', '20', '21']
# }
# keys = ['性别', '姓名', '年龄']
# values = ['19', '20']
# num = 0
# for key in keys:
#     for value in values:
#         try:
#             if value in a[key]:
#                 num += 1
#         except:
#             pass
a = 'dwadaw; dwadwa'
print(a.split(';').strip())