import json
import os
import random
import copy
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os, glob
import pandas as pd
from random import choice
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment

import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm
import json
import pickle
import os

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']


# 67->80
sr_to_ace_mapping = {}
for line in open(ROOT + "Datasets/M2E2/imSitu/ace_sr_mapping.txt", 'r'):
    fields = line.strip().split()
    sr_to_ace_mapping[fields[0]] = fields[2].replace('||', ':').replace('|', '-')

my_list = ['destroying', 'saluting', 'subduing', 'gathering', 'ejecting', 'marching', 'aiming', 'confronting',
           'bulldozing']
for m in my_list:
    if m in sr_to_ace_mapping:
        sr_to_ace_mapping.pop(m)
aaa = {
    "floating": "Movement:Transport",
    "leading": "Contact:Meet",
    "cheering": "Conflict:Demonstrate",
    "restraining": "Justice:Arrest-Jail",
    "bulldozing": "Conflict:Attack",
    "mourning": "Life:Die",
    "tugging": "Conflict:Attack",
    'signing': 'Contact:Meet',
    'colliding': "Conflict:Attack",
    'weighing': "Movement:Transport",
    'sleeping': "Life:Die",
    'falling': "Life:Die",
    'confronting': 'Contact:Meet',
    'gambling': 'Transaction:Transfer-Money',
    'pricking': 'Transaction:Transfer-Money'
}

for a in aaa:
    b = aaa[a]
    sr_to_ace_mapping[a] = b


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.3, 1.0),interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])



class ACE_Dataset_ARG(Dataset):
    def __init__(self, dataset, seq_len = 180):
        self.data = dataset
        self.data_length = len(dataset)
        self.seq_len = seq_len

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):

        data = self.data[index]
        sub_word_lens = [len(data[0])]
        max_sub_word_len = self.seq_len

        data_x, data_span, data_y = list(), list(), list()

        for i in range(len(data[1])):
            if data[1][i][0] < max_sub_word_len and data[1][i][1] < max_sub_word_len and data[1][i][2] < max_sub_word_len:
                data_span.append(data[1][i])
                data_y.append(data[2][i])
        if not data_span:
            data_span.append((0, 0))
            data_y.append(0)


        f = torch.LongTensor
        l = torch.LongTensor

        data_x = pad_sequence_to_length(data[0], max_sub_word_len)
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        images_path = data[-1]
        images = []
        for img in images_path:
            image = Image.open(img).convert('RGB')
            image = transform_test(image)
            images.append(image.unsqueeze(0))
        images = torch.cat(images, 0)

        dict = {'data_x': l(data_x),
                'bert_mask': bert_mask,
                'data_span': data_span,
                'data_y': data_y,
                'images': images}

        return dict

def load_conll_data(fpath):
    results = []

    words, tags_g, tags_p = list(), list(), list()
    for line in open(fpath, 'r'):
        line = line.strip()
        if len(line) == 0:
            if words:
                results.append([words, tags_g, tags_p])
                words, tags_g, tags_p = list(), list(), list()
        else:
            w = line.split(' ')[0]
            t1 = line.split(' ')[1]
            t2 = line.split(' ')[2]
            words.append(w)
            tags_g.append(t1)
            tags_p.append(t2)

    if words:
        results.append([words, tags_g, tags_p])
    return results

class M2E2_Dataset_ARG(Dataset):
    def __init__(self, dataset, event_result):

        self.data,self.trigger_result,self.sentence_id = list(),list(),list()
        trigger_result = load_conll_data(event_result)

        text_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_only_event.json').read())
        text_multimedia_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_multimedia_event.json').read())
        text_event.extend(text_multimedia_event)

        for data,event,original_label in zip(dataset,trigger_result,text_event):

            if not all(item == 'O' for item in event[2]):

                self.data.append(data)
                self.trigger_result.append(event)
                self.sentence_id.append(original_label['sentence_id'])

        self.data_length = len(self.data)


    def __len__(self):
        return self.data_length

    def __getitem__(self, index):

        data = self.data[index]
        trigger = self.trigger_result[index]
        d_x, span, _, ev, entity, image_path = data[0], data[1], data[2], data[3], data[4], data[5]
        f = torch.FloatTensor

        images = []
        for img in image_path:
            image = Image.open(img).convert('RGB')
            image = transform_test(image)
            images.append(image.unsqueeze(0))
        images = torch.cat(images, 0)


        dict = {'data_x': d_x,
                'span': span,
                'ev': ev,
                'entity': entity,
                'images': f(images),
                'event_result': trigger,
                'sentence_id':self.sentence_id[index]
                }

        return dict