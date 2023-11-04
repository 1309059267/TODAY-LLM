import json
import os

'''NER指标评估'''
with open('/home/sda/xuguangtao/Firefly-master/data/test/NER_answer.jsonl') as f:
    lines = f.readlines()
    golden_num = 0
    predict_num = 0
    correct_num = 0
    for line in lines:
        data = json.loads(line)
        label = data['label']
        split_list = label.split('\n')
        golden_entities = {}
        for element in split_list:
            element = element.strip()
            if '：' in element:
                entity_type, entities = element.split('：')
                entity_type = entity_type.strip()
                entities = entities.strip()
                if ';' in entities:
                    entities_raw = entities.split(';')
                    entities = []
                    for entity in entities_raw:
                        entities.append(entity.strip())
                else:
                    entities = [entities]
                golden_entities[entity_type] = entities
            else:
                pass
        for value in golden_entities.values():
            golden_num += len(value)
        
        answer = data['answer']
        split_list = answer.split('\n')
        predict_entities = {}
        for element in split_list:
            element = element.strip()
            if '：' in element:
                entity_type, entities = element.split('：')
                entity_type = entity_type.strip()
                entities = entities.strip()
                if ';' in entities:
                    entities_raw = entities.split(';')
                    entities = []
                    for entity in entities_raw:
                        entities.append(entity.strip())
                    entities = list(set(entities))
                else:
                    entities = [entities]
                predict_entities[entity_type] = entities
            else:
                pass
        for value in predict_entities.values():
            predict_num += len(value)
        
        for key in predict_entities.keys():
            values = predict_entities[key]
            for value in values:
                try: 
                    if value in golden_entities[key]:
                        correct_num += 1
                except:
                    pass
        
    print("数据集一共包含{}个实体".format(golden_num))
    print("模型共预测了{}个实体".format(predict_num))
    print("其中{}个正确的".format(correct_num))
    print("P:",correct_num/predict_num)
    print("R:",correct_num/golden_num)
    print("F1:",2/(1/(correct_num/predict_num)+1/(correct_num/golden_num)))    


'''RE指标评估'''
with open('/home/sda/xuguangtao/Firefly-master/data/test/RE_answer.jsonl') as f:
    lines = f.readlines()
    golden_num = 0
    predict_num = 0
    correct_num = 0
    for line in lines:
        data = json.loads(line)
        label = data['label']
        split_list = label.split('\n')
        golden_entities = {}
        for element in split_list:
            element = element.strip()
            if ':' in element:
                entity_type, entities = element.split(':')
                entity_type = entity_type.strip()
                entities = entities.strip()
                if ';' in entities:
                    entities_raw = entities.split(';')
                    entities = []
                    for entity in entities_raw:
                        entities.append(entity.strip())
                else:
                    entities = [entities]
                golden_entities[entity_type] = entities
            else:
                pass
        for value in golden_entities.values():
            golden_num += len(value)
        
        answer = data['answer']
        split_list = answer.split('\n')
        predict_entities = {}
        for element in split_list:
            element = element.strip()
            if ':' in element:
                entity_type, entities = element.split(':')
                entity_type = entity_type.strip()
                entities = entities.strip()
                if ';' in entities:
                    entities_raw = entities.split(';')
                    entities = []
                    for entity in entities_raw:
                        entities.append(entity.strip())
                    entities = list(set(entities))
                else:
                    entities = [entities]
                predict_entities[entity_type] = entities
            else:
                pass
        for value in predict_entities.values():
            predict_num += len(value)
        
        for key in predict_entities.keys():
            values = predict_entities[key]
            for value in values:
                try: 
                    if value in golden_entities[key]:
                        correct_num += 1
                except:
                    pass
           
    print("数据集一共包含{}个关系三元组".format(golden_num))
    print("模型共预测了{}个关系三元组".format(predict_num))
    print("其中{}个正确的".format(correct_num))
    print("P:",correct_num/predict_num)
    print("R:",correct_num/golden_num)
    print("F1:",2/(1/(correct_num/predict_num)+1/(correct_num/golden_num)))  

 


'''下面是宽松版NER和RE指标评估代码，不需要看'''
# with open('/home/sda/xuguangtao/Firefly-master/data/test/RE_answer.jsonl') as f:
#     lines = f.readlines()
#     golden_num = 0
#     predict_num = 0
#     correct_num = 0
#     for line in lines:
#         data = json.loads(line)
#         label = data['label']
#         split_list = label.split('\n')
#         golden_entities = {}
#         labels_num = 0
#         for element in split_list:
#             element = element.strip()
#             if ':' in element:
#                 entity_type, entities = element.split(':')
#                 entity_type = entity_type.strip()
#                 entities = entities.strip()
#                 if ';' in entities:
#                     entities_raw = entities.split(';')
#                     entities = []
#                     for entity in entities_raw:
#                         entities.append(entity.strip())
#                     entities = list(set(entities))
#                 else:
#                     entities = [entities]
#                 labels_num += len(entities)


#                 new_entities = []
#                 for entity in entities:
#                     try:
#                         A, B = entity[1:-1].split(', ')
#                         new_entity = '[{}, {}]'.format(B, A)
#                         new_entities.append(new_entity)
#                     except:
#                         pass
#                 entities += new_entities
#                 golden_entities[entity_type] = entities
#             else:
#                 pass
        
#         golden_num += labels_num

        
#         answer = data['answer']
#         split_list = answer.split('\n')
#         predict_entities = {}
#         for element in split_list:
#             element = element.strip()
#             if ':' in element:
#                 entity_type, entities = element.split(':')
#                 entity_type = entity_type.strip()
#                 entities = entities.strip()
#                 if ';' in entities:
#                     entities_raw = entities.split(';')
#                     entities = []
#                     for entity in entities_raw:
#                         entities.append(entity.strip())
#                     entities = list(set(entities))
#                 else:
#                     entities = [entities]
#                 predict_entities[entity_type] = entities
#             else:
#                 pass
#         for value in predict_entities.values():
#             predict_num += len(value)
        
#         for key in predict_entities.keys():
#             values = predict_entities[key]
#             for value in values:
#                 try: 
#                     if value in golden_entities[key]:
#                         correct_num += 1
#                 except:
#                     pass
           
#     print("数据集一共包含{}个关系三元组".format(golden_num))
#     print("模型共预测了{}个关系三元组".format(predict_num))
#     print("其中{}个正确的".format(correct_num))
#     print("P:",correct_num/predict_num)
#     print("R:",correct_num/golden_num)
#     print("F1:",2/(1/(correct_num/predict_num)+1/(correct_num/golden_num)))  
