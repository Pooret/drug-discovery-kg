# imports
import xml.etree.ElementTree as ET
import json
import os
import copy
import glob
import re
import random

def parse_ddi_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    for sentence in root.findall('.//sentence'):
        text = sentence.attrib['text']
        text_id = sentence.attrib['id']
        pairs = []
        
        for pair in sentence.findall('.//pair'):
            e1_id = pair.attrib['e1']
            e2_id = pair.attrib['e2']
            ddi = pair.attrib['ddi'] == 'true'
            ddi_type = pair.attrib.get('type', 'unknown')
            pairs.append((e1_id, e2_id, ddi, ddi_type))
        
        entities = {}
        for entity in sentence.findall('.//entity'):
            char_offset_str = entity.attrib['charOffset']
            entity_spans = char_offset_str.split(';')
            spans = []
            for span in entity_spans:
                start, end = map(int, span.split('-'))
                spans.append((start, end))
            if spans:
                entities[entity.attrib['id']] = {
                    'text': entity.attrib['text'],
                    'type': entity.attrib['type'],
                    'char_offset': spans
                }
            else:
                entities[entity.attrib['id']] = {
                    'text': entity.attrib['text'],
                    'type': entity.attrib['type'],
                    'char_offset': None
                }
        
        for e1_id, e2_id, ddi, ddi_type in pairs:
            data.append({
                'filename': filename,
                'sentence': text,
                'sent_id': text_id,
                'entity1': {
                    'text': entities[e1_id]['text'],
                    'type': entities[e1_id]['type'],
                    'char_offset': entities[e1_id]['char_offset']
                },
                'entity2': {
                    'text': entities[e2_id]['text'],
                    'type': entities[e2_id]['type'],
                    'char_offset': entities[e2_id]['char_offset']
                },
                'ddi': ddi,
                'type': ddi_type,
                'all_ents': entities
            })
    
    return data

def preprocess_data(*corpus_paths, parse_function):
    """
    Preprocess corpora of XML files for entity extraction with provided paths and parsing function.
    Returns a list of sentences with entity/relationship pairs
    """
    data = []
    for corpus_path in corpus_paths:
        for file in os.listdir(corpus_path):
            if file.endswith('.xml'):
                file_path = os.path.join(corpus_path, file)
                data.extend(parse_function(file_path))
    
    return data

def extract_ner_data(input_data):
    ner_data = []
    data = copy.deepcopy(input_data)
    # Iterate through each instance in the data
    for instance in data:
        input_data = {}
        input_data['sentence'] = instance['sentence']
        input_data['entities'] = []

        # Check if there is a DDI (drug-drug interaction) in the instance
        if instance['ddi']:
            char_offset1 = instance['entity1']['char_offset']
            char_offset2 = instance['entity2']['char_offset']
        else:
            # Set dummy values if there is no DDI
            char_offset1, char_offset2 = -1, -1

        # Iterate through all entities in the instance
        for value in instance['all_ents'].values():
            # Check if the entity has a relation (based on char_offset)
            if (value['char_offset'] == char_offset1) or (value['char_offset'] == char_offset2):
                # print(value['char_offset'], char_offset1, char_offset2)
                # Update the entity type to include the relation type
                relation_type = value['type']
#                 value['type'] = f"{value['type']}-{instance['type']}"
                relation_entity = value.copy()
                relation_entity['type'] = relation_type + f"-{instance['type']}"
                input_data['entities'].append(relation_entity)
            else:
            # Append the entity to the input_data
                input_data['entities'].append(value)

        # Append the input_data to the ner_data
        ner_data.append(input_data)
    
    return ner_data

def extract_ddi_data(data):
    ddi_data = []
    for instance in data:
        input_data = {}
        input_data['relations'] = {}
        
        input_data['sentence'] = instance['sentence']
        input_data['relations']['entity1'] = instance['entity1']
        input_data['relations']['entity2'] = instance['entity2']
        input_data['relations']['ddi'] = instance['ddi']
        input_data['relations']['type'] = instance['type']
        
        ddi_data.append(input_data)
    return ddi_data

def create_train_val_split(data, val_split=0.1):
    """
    shuffles and splits the data 
    """
    random.shuffle(data)
    val_size = int(len(data) * val_split)
    train_data = data[:-val_size]
    val_data = data[-val_size:]
    return train_data, val_data

def generate_conll_format(data, return_labels=False):
    sentence = data['sentence']
    entities = data['entities']

    labels = ['O'] * len(sentence)
 

    # Assign labels using character offsets
    for entity in entities:
        span = entity['char_offset']
        counter = 0
        flag = False
        if len(span) > 0:
            flag = True
        for start, end in span:
            if start < len(sentence) and end <= len(sentence):
                if len(labels[start].split("-")) < 3:
                    labels[start] = f"B-{entity['type']}"
                    if flag:
                        counter += 1
                        if counter == 2:
                            labels[start] = f"I-{entity['type']}"
                for i in range(start + 1, end + 1):
                    if len(labels[i].split("-")) < 3:
                        labels[i] = f"I-{entity['type']}"
    
    # Tokenize the sentence using regex to split on whitespace or special characters
    pattern = r"(\w+|\S)"
    tokens = [match.group() for match in re.finditer(pattern, sentence)]

    # Combine tokens and labels into the CoNLL format
    conll_format = {}
    conll_format['tokens'] = []
    conll_format['tags'] = []
    token_start = 0
    for token in tokens:
        token_start = sentence.find(token, token_start)
        token_end = token_start + len(token) - 1
        token_label = labels[token_start:token_end + 1]
        
        # Find the main label by checking if there are any "B-" or "I-" labels in the token_label list
        main_label = next((label for label in token_label if label.startswith("B-") or label.startswith("I-")), 'O')
        
        conll_format['tokens'].append(token)
        conll_format['tags'].append(main_label)
        token_start = token_end + 1
    
    if return_labels:
        return conll_format, labels
    else:
        return conll_format
    
def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, max_length=512):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword.
    """

    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):

        # tokenize word and count # of subword tokens
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # add tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # add label and multiply by subword length
        labels.extend([label] * n_subwords)

    tokenized_sentence += [tokenizer.pad_token] * (max_length - len(tokenized_sentence))
    labels += ['O'] * (max_length - len(labels))
    #tokenized_sentence.extend(tokenizer.pad_token * (max_length - len(tokenized_sentence)))
    #print(tokenized_sentence)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    attention_mask = [1 if token != tokenizer.pad_token else 0 for token in tokenized_sentence]
    #return input_ids, attention_mask, labels
    return tokenized_sentence, labels


ner_sample = {'sentence': 'H1 and H2 Blockers - Although not reported, L-histidine, via its metabolism to histamine, might decrease the efficacy of H1 and H2 blockers.',
  'entities': [{'text': 'H1 Blockers',
    'type': 'group',
    'char_offset': [(0, 1), (10, 17)]},
   {'text': 'H2 Blockers', 'type': 'group', 'char_offset': [(7, 17)]},
   {'text': 'L-histidine', 'type': 'drug-effect', 'char_offset': [(44, 54)]},
   {'text': 'H1 blockers',
    'type': 'group-effect',
    'char_offset': [(121, 122), (131, 138)]},
   {'text': 'H2 blockers', 'type': 'group', 'char_offset': [(128, 138)]}]}

result = generate_conll_format(ner_sample)
