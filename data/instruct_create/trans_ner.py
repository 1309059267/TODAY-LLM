import json
import jsonlines
import random

CMeEE = [
    "找出指定的实体：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]",
    "找出指定的实体：\n[INPUT_TEXT]\n实体类型选项：[LIST_LABELS]",
    "找出句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]实体是什么？",
    "生成句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]",
    "下面句子中的[LIST_LABELS]实体有哪些？\n[INPUT_TEXT]",
    "实体抽取：\n[INPUT_TEXT]\n选项：[LIST_LABELS]",
    "医学实体识别：[INPUT_TEXT]\n实体选项：[LIST_LABELS]",
    "在下述文本中标记出医学实体：\n[INPUT_TEXT]\n实体类型：[LIST_LABELS]",
    "找出下面文本中的[LIST_LABELS]实体：“[INPUT_TEXT]”",
  ]

CMeEEAugment = [
    "找出指定的实体：\n[INPUT_TEXT]\n类型选项：[LIST_LABELS]",
    "找出指定的实体：\n[INPUT_TEXT]\n实体类型选项：[LIST_LABELS]",
    "找出句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]",
    "[INPUT_TEXT]\n问题：句子中的[LIST_LABELS]实体是什么？",
    "生成句子中的[LIST_LABELS]实体：\n[INPUT_TEXT]",
    "下面句子中的[LIST_LABELS]实体有哪些？\n[INPUT_TEXT]",
    "实体抽取：\n[INPUT_TEXT]\n选项：[LIST_LABELS]",
    "医学实体识别：\n[INPUT_TEXT]\n实体选项：[LIST_LABELS]",
    "在下面的文本中找出医疗命名实体：\n[INPUT_TEXT]\n请从以下类型中选择医疗命名实体：[LIST_LABELS]",
    "请从以下文本中提取医疗命名实体：\n[INPUT_TEXT]\n医疗命名实体的类型包括：[LIST_LABELS]",
    "识别以下文本中的医疗命名实体：\n[INPUT_TEXT]\n请标注以下类型的医疗命名实体：[LIST_LABELS]",
    "请从以下句子中找出[LIST_LABELS]实体：\n[INPUT_TEXT]",
    "[INPUT_TEXT]中包含了哪些[LIST_LABELS]实体？请列举出来。",
    "在给定的文本[INPUT_TEXT]中，请标出所有的[LIST_LABELS]实体。",
    "[INPUT_TEXT]\n问题：请标出句子中的[LIST_LABELS]实体。",
    "[INPUT_TEXT]\n问题：你能识别出句子中的哪些[LIST_LABELS]实体吗？",
    "[INPUT_TEXT]\n问题：根据句子内容，找出其中的[LIST_LABELS]实体。",
    "请识别以下医学文本中的实体：\n[INPUT_TEXT]\n实体类型：[LIST_LABELS]",
    "在下述文本中标记出医学实体：\n[INPUT_TEXT]\n可识别的实体有：[LIST_LABELS]",
    "医学命名实体识别任务：\n请从下面的文本中提取医疗实体：\n[INPUT_TEXT]\n实体类型包括：[LIST_LABELS]",
    "请标出以下句子中的医疗实体类型为[LIST_LABELS]的实体：\n[INPUT_TEXT]",
    "请识别并列举出以下文本中属于[LIST_LABELS]类型的医疗实体：\n[INPUT_TEXT]",
    "在下面的句子中找出所有关于[LIST_LABELS]的医疗实体：\n[INPUT_TEXT]",
    "在下述文本中标记出医学实体：\n[INPUT_TEXT]\n实体类型：[LIST_LABELS]",
    "找出下面文本中的[LIST_LABELS]实体：“[INPUT_TEXT]”",
]

LIST_LABELS = "微生物类，疾病，药物，医疗程序，医疗设备，临床表现，科室，身体，医学检验项目"

def trans_ner_instruct(original_file, instruct_file,entity_type_map = None):
    for document in original_file:
        document_output = dict()
        d_id = document['id']
        passages = document['passages']
        entities = document['entities']
        text = CMeEEAugment[random.randint(0,24)]
        entity_dict = dict()
        for passage in passages:
            # 将"[LIST_LABELS]"替换为LIST_LABELS
            text = text.replace("[LIST_LABELS]",LIST_LABELS)
            # 将"[INPUT_TEXT]"替换为输入文本
            text = text.replace("[INPUT_TEXT]",passage['text'][0])
        for entity in entities:
            ent_type = entity['type']
            if entity_type_map is not None:
                ent_type = entity_type_map[ent_type]
            ent_text = entity['text'][0]
            if ent_type not in entity_dict.keys():
                entity_dict[ent_type] = [ent_text]
            else:
                if ent_text not in entity_dict[ent_type]:
                    entity_dict[ent_type].append(ent_text)
        answer = str()
        for key, value in entity_dict.items():
            answer += key + "：" 
            for v in value:
                answer += v + "; " 
            answer = answer[:-2] + "\n"
        assistant = json.dumps(entity_dict,ensure_ascii = False)
        document_output = {
            "conversation_id" : int(d_id),
            "category" : "NER",
            "conversation" : [{"human" : text,
                               "assistant" : answer}]
        }  
        json.dump(document_output, instruct_file, ensure_ascii=False)
        instruct_file.write('\n')
cmeee_map =  {
    "dis":"疾病",
    "sym":"临床表现",
    "pro":"医疗程序",
    "equ":"医疗设备",
    "dru":"药物",
    "ite":"医学检验项目",
    "bod":"身体",
    "dep":"科室",
    "mic":"微生物类"
    }
imcs_map = {
    "Symptom":"症状",
    "Drug":"药品名",
    "Drug_Category":"药物类别",
    "Medical_Examination":"检查",
    "Operation":"操作"
}

with jsonlines.open('origin.jsonl') as original_file, \
      open('CMeEE-V2_train_instruct_augment.jsonl','w',encoding='utf-8')  as output_file:
    trans_ner_instruct(original_file = original_file,instruct_file = output_file,entity_type_map = None)
