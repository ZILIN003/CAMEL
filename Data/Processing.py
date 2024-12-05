import json
import pickle
from re import L
from PIL import Image

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import os.path
import config
import random
import glob

ace_path = 'Datasets/M2E2/ace/ACE/anno_event_json2'
ROOT = ""
ace_files = 'Datasets/M2E2/voa/'

tokenizer = config.tokenizer
tag2idx = config.tag2idx



def read_m2e2():
    file1 = ace_files+'m2e2_annotations/text_only_event.json'
    file2 = ace_files+'m2e2_annotations/text_multimedia_event.json'

    dict={}

    for f in [file1, file2]:
        xxx = json.loads(open(f).read())
        for x in xxx:
            sentence_id = x['sentence_id']
            sentence = x['words']
            event = x['golden-event-mentions']
            image = x['image']
            image = [ace_files+'m2e2_rawdata/image/image/' + t for t in image]
            image = list(filter(lambda x: os.path.exists(x), image))
            for i in range(5):
                image.append(f"/Event/BLIP/MEE/textualEE/ACE_Image/black.jpg")
            image = image[:5]

            entity = x['golden-entity-mentions']
            if sentence_id not in dict:
                dict[sentence_id]=[sentence, event, image, entity]
            else:
                dict[sentence_id][1].extend(event)

    temp=list(dict.values())

    return temp

def build_bert_examples_m2e2(result, Flag=True):
    # construct examples for event type detection when flag=True
    examples = []

    for doc in result:
        sen, ev, image, entity = doc
        sen += ['[SEP]']
        labels = ['O'] * len(sen)
        for e in ev:
            i, t = e['trigger']['start'], e['event_type'].split(':')[1]
            labels[i] = t

        subword_ids, spans, label_ids = _to_bert_examples(sen, labels)
        if Flag:
            examples.append([subword_ids, spans, label_ids, image, sen, labels])
        else:
            examples.append([subword_ids, spans, label_ids, ev, entity, image])

    return examples


def _to_bert_examples(words, labels):
    subword_ids = list()
    spans = list()
    label_ids = list()
    for word in words:
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        spans.append([s, e])

    for label in labels:
        label_ids.append(tag2idx.get(label, 0))

    return subword_ids, spans, label_ids


def read_ace_event():
    result = []
    data = []
    file_name = ROOT + 'Datasets/M2E2/ace/'
    data += json.load(open(file_name + 'JMEE_train_filter_no_timevalue.json', 'r'))
    data += json.load(open(file_name + 'JMEE_dev_filter_no_timevalue.json', 'r'))
    data += json.load(open(file_name + 'JMEE_test_filter_no_timevalue.json', 'r'))
    for line_data in data:
        sentence, event, words = (
            line_data['sentence'], line_data['golden-event-mentions'], line_data['words'])
        result.append([sentence, event, words])
    return result

def build_bert_examples_event(result):
    examples = []


    img_path = '/home/duzilin/Code/StableDiffusion/dif2-1'
    all_files = os.listdir(img_path)
    all_files.remove('M2E2')
    num=0
    for i, doc in enumerate(result):
        sentence, event, words = doc
        img_list = []
        img_name1 = "%05d-0-sub.png" % i

        if img_name1 in all_files:
            for index in range(4):
                temp_path = os.path.join(img_path, "%05d-%d-sub.png" % (i, index))
                if os.path.exists(temp_path):
                    img_list.append(temp_path)
                    num+=1

        if len(img_list) != 5:
            ramdom_list = random.sample(all_files, 5 - len(img_list))
            list(map(lambda x: os.path.join(img_path, x),ramdom_list))
            img_list.extend(list(map(lambda x: os.path.join(img_path, x),ramdom_list)))

        labels = ['O'] * len(words)
        words += ['[SEP]']

        for id, event_item in enumerate(event):
            start_index = event_item['trigger']['start']
            end_index = event_item['trigger']['end']
            type = event_item['event_type'].split(':')[1]
            if type in config.event_type:
                for i in range(start_index, end_index):
                    labels[i] = type

        subword_ids, spans, label_ids = _to_bert_examples(words, labels)

        examples.append([subword_ids, spans, label_ids, img_list, words, labels])

    print("{} images are added.".format(num))

    return examples


def read_ace_argument():
    result = []
    data = []
    file_name = ROOT + 'Datasets/M2E2/ace/'
    data += json.load(open(file_name + 'JMEE_train_filter_no_timevalue.json', 'r'))
    data += json.load(open(file_name + 'JMEE_dev_filter_no_timevalue.json', 'r'))
    data += json.load(open(file_name + 'JMEE_test_filter_no_timevalue.json', 'r'))
    for line_data in data:
        sentence, event, words, entity = (
            line_data['sentence'], line_data['golden-event-mentions'], line_data['words'], line_data['golden-entity-mentions'])
        result.append([sentence, event, words, entity])
    return result

def build_bert_examples_argument(result,event_examples):
    examples = []
    for index, doc in enumerate(result):

        sentence, event, words, entity = doc

        labels = ['O'] * len(words)
        words += ['[SEP]']
        subword_ids, spans, label_ids = _to_bert_examples(words, labels)

        pairs = list()
        labels = list()

        for event_item in event:
            i = event_item['trigger']['start']
            args = event_item['arguments']
            for entity_item in entity:
                flag = False
                for a in args:
                    if entity_item['text'] == a['text']:
                        if a['role'] in config.tag2idx_role:
                            pairs.append((spans[i][0], spans[a['start']][0], spans[a['end']][-1]))
                            labels.append(config.tag2idx_role[a['role']])
                            flag = True

                if not flag:
                    pairs.append((spans[i][0], spans[entity_item['start']][0], spans[entity_item['end']][-1]))
                    labels.append(0)

        if pairs:
            examples.append([subword_ids, pairs, labels, words, event_examples[index][3]])

    return examples




if __name__ == '__main__':

    # ACE event dataset
    result = read_ace_event(ace_path)
    event_examples = build_bert_examples_event(result)
    with open("./data/ace_event.pkl", "wb") as f:
        pickle.dump(event_examples, f)

    # ACE arguement dataset
    result = read_ace_argument()
    arg_examples = build_bert_examples_argument(result,event_examples)
    with open("./data/new_img/ace_arg.pkl", "wb") as f:
        pickle.dump(arg_examples, f)


    # M2E2 dataset
    result = read_m2e2()
    examples = build_bert_examples_m2e2(result)
    with open("./data/m2e2_event.pkl", "wb") as f:
        pickle.dump(examples, f)
    examples = build_bert_examples_m2e2(result,False)
    with open("./data/m2e2_arg.pkl", "wb") as f:
        pickle.dump(examples, f)

