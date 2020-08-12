import json
# def type_transformer(entity):

f = open('train.txt', 'w')
with open('train.json', 'r') as fp:
    sentences = []
    for line in fp.readlines():
        sentence = json.loads(line)
        labels = sentence['label']
        text = sentence['text']
        target = []
        for i in range(len(text)):
            target.append('O')
        for key in labels:
            entity_type = key
            entity_dict = labels[key]    # e.g. {'叶老桂': [[9, 11]]}
            for entity_name in entity_dict:
                entity_start_index = entity_dict[entity_name][0][0]
                entity_end_index = entity_dict[entity_name][0][1]
                entity_length = entity_end_index - entity_start_index + 1
                target[entity_start_index] = 'B-'+str(entity_type)
                if entity_length != 1:
                    for i in range(entity_start_index+1, entity_end_index+1):
                        target[i] = 'I-'+str(entity_type)
        f.write(str(target))
        f.write('\n')
f.close()