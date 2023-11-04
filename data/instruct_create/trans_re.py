import json
import random
from collections import defaultdict

CMeIEV2_prompt = [
    "给定实体间的关系[LIST_LABELS]，抽取下面文本中具有这些关系的实体对：\n[INPUT_TEXT]",
    "找出句子中的具有[LIST_LABELS]这些关系类型的实体对：\n[INPUT_TEXT]",
    "[INPUT_TEXT]\n上面句子中关系类型为[LIST_LABELS]的实体对有哪些？",
    "给出下面句子中关系类型为[LIST_LABELS]的实体对。\n[INPUT_TEXT]",
    "请找出下述文本中的实体关系三元组：\n[INPUT_TEXT]\n关系类型：[LIST_LABELS]。",
    "请根据下面的文本提取出具有[LIST_LABELS]这些关系类型的实体对。\n[INPUT_TEXT]",
    "根据下述文本，提取出具有[LIST_LABELS]这些关系的实体对。\n[INPUT_TEXT]",
    "从以下文本中抽取具有[LIST_LABELS]这些关系的实体对：\n[INPUT_TEXT]",
    "在下面的句子中，提取出关系类型为[LIST_LABELS]的实体对：\n[INPUT_TEXT]",
    "[INPUT_TEXT]\n问题：在上述文本中，找出关系类型为[LIST_LABELS]的实体对。",
    "请给出句子中关系类型为[LIST_LABELS]的实体对：\n[INPUT_TEXT]",
    "从下面的句子中提取出关系类型为[LIST_LABELS]的实体对：\n[INPUT_TEXT]",
    "在下述文本中，找出关系类型为[LIST_LABELS]的实体对：\n[INPUT_TEXT]",
    "关系类型：[LIST_LABELS]。\n输入文本：[INPUT_TEXT]\n找出上述文本中所有的实体关系对。",
    "从下面文本中抽取实体关系：\n[INPUT_TEXT]\n关系类型：[LIST_LABELS]。",
    "实体间关系抽取：\n[INPUT_TEXT]\n关系类型：[LIST_LABELS]。",
    "[INPUT_TEXT]\n请找出上述文本中的关系三元组。\n关系类型为[LIST_LABELS]。",
    "[INPUT_TEXT]\n找出上面文本中所有的关系三元组。\n关系类型：[LIST_LABELS]。",
]

rel={
        "多发地区": "",
        "就诊科室": "",
        "放射治疗": "",
        "临床表现": "",
        "相关（症状）": "",
        "发病年龄": "",
        "死亡率": "",
        "病理分型": "",
        "外侵部位": "",
        "辅助治疗": "",
        "病因": "",
        "发病机制": "",
        "治疗后症状": "",
        "转移部位": "",
        "辅助检查": "",
        "传播途径": "",
        "预后状况": "",
        "影像学检查": "",
        "内窥镜检查": "",
        "预防": "",
        "多发季节": "",
        "风险评估因素": "",
        "鉴别诊断": "",
        "相关（转化）": "",
        "发病部位": "",
        "病史": "",
        "发病率": "",
        "手术治疗": "",
        "预后生存率": "",
        "实验室检查": "",
        "相关（导致）": "",
        "筛查": "",
        "高危因素": "",
        "组织学检查": "",
        "药物治疗": "",
        "同义词": "",
        "多发群体": "",
        "并发症": "",
        "发病性别倾向": "",
        "化疗": "",
        "遗传因素": "",
        "病理生理": "",
        "阶段": "",
        "侵及周围组织转移的症状": ""
    }

relation_total = list(rel.keys())
print(len(relation_total))

def trans_re_instruct_chinese(original_file, instruct_file,entity_type_map = None, dataset = None,get_des_flag = False,des_file = None):
    all_ent_type_set = set()
    all_rel_type_set = set()
    examples = list()
    des_dict = {
            "task_des": "",
            "ent_type": {
            },
            "rel_type": {
            },
            "example": [
            ],
            "eval": ["Micro-F1"]
        }

    for d_id, original_data in enumerate(original_file.readlines()):
        rel_total = []
        original_data = json.loads(original_data)
        passages = original_data['passages']
        entities = original_data['entities']
        relations = original_data['relations']
        text = str()
        entity_dict = dict()
        id_to_ent = dict()
        for passage in passages:
            text += passage['text'][0]
        for entity in entities:
            ent_id = entity['id']
            ent_type = entity['type']
            all_ent_type_set.add(ent_type)
            if entity_type_map is not None:
                ent_type = entity_type_map[ent_type]
            ent_text = entity['text'][0]
            if ent_id not in id_to_ent.keys():
                id_to_ent[ent_id] = ent_text
            if ent_type not in entity_dict.keys():
                entity_dict[ent_type] = [ent_text]
            else:
                if ent_text not in entity_dict[ent_type]:
                    entity_dict[ent_type].append(ent_text)

        example_flag = False
        triplet = []
        for relation in relations:
            head_ent_id = relation["arg1_id"]
            tail_ent_id = relation["arg2_id"]
            head_ent_text = id_to_ent[head_ent_id].strip("'")
            tail_ent_text = id_to_ent[tail_ent_id].strip("'")
            re_type = relation["type"]
            if re_type not in rel_total:
                rel_total.append(re_type)
            if re_type not in all_rel_type_set:
                all_rel_type_set.add(re_type)
                example_flag = True
            if (head_ent_text,re_type,tail_ent_text) not in triplet:
                triplet.append((head_ent_text, re_type, tail_ent_text))

        answer = str()
        for re in rel_total:
            one_relation = str(re) + "："
            for h,r,t in triplet:
                if re == r:
                   one_relation += str([h,t]).replace("'","") + "; "
            one_relation = one_relation[:-2] + "\n"
            answer += one_relation

        list_label = ','.join(map(str, relation_total))
        text_init = random.choice(CMeIEV2_prompt)
        text_final = text_init.replace('[INPUT_TEXT]',text).replace('[LIST_LABELS]',list_label)

        document_output = {
            "conversation_id": d_id,
            "category": "RE",
            "conversation": [{"human": text_final,
                              "assistant": answer}]
        }
        # assistant = str(answer)
        # examples.append(text + " -> " + assistant)
        # if des_file is not None and example_flag:
        #     des_dict["example"].append(text + " -> " + assistant)
        json.dump(document_output, instruct_file, ensure_ascii=False)
        instruct_file.write('\n')

    # if des_file is not None:
    #     for ent_type in all_ent_type_set:
    #         des_dict["ent_type"][ent_type] = ""
    #     for rel_type in all_rel_type_set:
    #         des_dict["rel_type"][rel_type] = ""
    #     example_index = 0
    #     while (len(des_dict["example"]) < 4):
    #         example = examples[example_index]
    #         if example not in des_dict["example"]:
    #             des_dict["example"].append(example)
    #         example_index += 1
    #     json.dump(des_dict, des_file, ensure_ascii=False, indent=4)



with open('/home/sda/xuguangtao/Firefly-master/data/instruct_create/cmeie_v2-train.jsonl','r',encoding='utf-8') as original_file, \
      open('/home/sda/xuguangtao/Firefly-master/data/instruct_create/cmeie_v2-train_instruct.jsonl','w',encoding='utf-8')  as output_file:
    trans_re_instruct_chinese(original_file = original_file,instruct_file = output_file)