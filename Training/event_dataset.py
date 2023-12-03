import json
import os
import random
import copy
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os, glob
from random import choice
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment

import sys
import torch
import pickle
import random

from tqdm import tqdm
import json
import pickle
import os

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']


# 67->80
sr_to_ace_mapping = {}
for line in open("Datasets/M2E2/imSitu/ace_sr_mapping.txt", 'r'):
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

star_verb_list = list(sr_to_ace_mapping.keys())
imsitu = json.load(open("Datasets/M2E2/imSitu/imsitu_space.json"))
verbs_org = imsitu["verbs"]
verb_list = [v for v in verbs_org]

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
max_val = np.array([(1. - mean[0]) / std[0],
                    (1. - mean[1]) / std[1],
                    (1. - mean[2]) / std[2],
                    ])

min_val = np.array([(0. - mean[0]) / std[0],
                    (0. - mean[1]) / std[1],
                    (0. - mean[2]) / std[2],
                    ])

eps_size = np.array([abs((1. - mean[0]) / std[0]) + abs((0. - mean[0]) / std[0]),
                     abs((1. - mean[1]) / std[1]) + abs((0. - mean[1]) / std[1]),
                     abs((1. - mean[2]) / std[2]) + abs((0. - mean[2]) / std[2]),
                     ])


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224,scale=(0.7, 1.0),interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2, 5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])

Caption='blip'

class imSitu_dataset(Dataset):
    def __init__(self, type, vote):
        ann_file = 'Datasets/M2E2/imSitu/'
        if type == 'train':
            self.data = json.load(open(os.path.join(ann_file,"train.json"),'r'))
            dev_data = json.load(open(os.path.join(ann_file, "dev.json"), 'r'))
            self.data.update(dev_data)

            # # dynamic
            # no_event = {}
            # event = {}
            # for k,v in copy.deepcopy(self.data).items():
            #     if v['verb'] not in sr_to_ace_mapping:
            #         no_event[k] = v
            #     else:
            #         event[k] = v
            # # 14306
            # sample_list = random.sample(no_event.keys(), 14306)
            # for item in sample_list:
            #     event[item] = no_event[item]
            # self.data = event

            # # static
            # # flag = 0
            # # for k,v in copy.deepcopy(self.data).items():
            # #     if v['verb'] not in sr_to_ace_mapping:
            # #         if flag>=14300:
            # #             self.data.pop(k)
            # #         flag+=1

            self.transform = transform_train
            with open('/home/duzilin/Code/Event/BLIP/Unify/caption/'+Caption+'_train.json', 'r') as f:
                self.caption = json.load(f)

        else:
            self.data  = json.load(open(os.path.join(ann_file, "test.json"), 'r'))
            self.transform = transform_test
            with open('/home/duzilin/Code/Event/BLIP/Unify/caption/'+Caption+'_val.json', 'r') as f:
                self.caption = json.load(f)

        self.nouns = json.load(open(os.path.join(ann_file, "imsitu_space.json"), 'r'))['nouns']
        self.imgs_names = list(self.data.keys())
        self.img_dir = ann_file+'of500_images_resized/'
        self.verbs = verb_list
        self.vote = vote


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.imgs_names[index]
        annotations = self.data[img_name]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        img = self.transform(img)
        verb= annotations['verb']

        if self.vote:
            cls = self.verbs.index(verb)
        else:
            v = self.caption[img_name]['verb']
            if v in sr_to_ace_mapping:
                cls = LABELS_list.index(sr_to_ace_mapping[self.caption[img_name]['verb']])
            else:
                cls = LABELS_list.index('None')
        cap = self.caption[img_name]['cap']

        return img, cls, cap



class ACE_Dataset(Dataset):
    def __init__(self, dataset, seq_len = 180):
        self.data = dataset
        self.data_length = len(dataset)
        self.seq_len = seq_len

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):

        data_item = self.data[index]

        sub_word_lens = [len(data_item[0])]
        max_sub_word_len = self.seq_len

        original_sentence_len = [len(data_item[2])]
        max_original_sentence_len = self.seq_len

        data_x = data_item[0]
        data_span = data_item[1]
        data_y = data_item[2]
        text = ' '.join(data_item[-2])
        images_path = data_item[3]
        images = []
        for img in images_path:
            image = Image.open(img).convert('RGB')
            image = transform_test(image)
            images.append(image.unsqueeze(0))
        images = torch.cat(images, 0)

        words = data_item[-2]
        labels = data_item[-1]

        f = torch.FloatTensor
        l = torch.LongTensor

        data_x = pad_sequence_to_length(data_x, max_sub_word_len)
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        default_y = -1
        data_y = pad_sequence_to_length(data_y, max_original_sentence_len, default_value=lambda: default_y)

        sequence_mask = get_mask_from_sequence_lengths(f(original_sentence_len), max_original_sentence_len)

        data_span_tensor = np.zeros((max_original_sentence_len, 2), dtype=int)

        temp = data_span[:max_original_sentence_len]

        for elem in temp:
            if elem[0] >= self.seq_len:
                elem[0] = self.seq_len - 1
            if elem[1] >= self.seq_len:
                elem[1] = self.seq_len - 1

        data_span_tensor[:len(temp), :] = temp

        dict = {'data_x':l(data_x),
                'bert_mask':bert_mask,
                'data_span_tensor':l(np.array(data_span_tensor)),
                'sequence_mask':sequence_mask,
                'data_y':l(data_y),
                'images':images,
                'text':text,
                'words':words,
                'labels':labels,
                'images_num':len(images_path)}

        return dict

class m2e2_img_Dataset(Dataset):
    def __init__(self):
        self.labels = []
        self.images = []
        event_dict = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_only_event.json').read())
        event_dict1 = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_multimedia_event.json').read())
        event_dict.update(event_dict1)
        image_dir = 'Datasets/M2E2/voa/m2e2_rawdata/image/image'
        all_images = [image_dir + '/' + f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.img_names = []
        for img in all_images:
            img_name = img[img.rfind('/')+1:]
            img_name = img_name[:img_name.rfind('.')]
            if img_name in event_dict:
                txt_label = event_dict[img_name]['event_type']
                label = txt_label
            else:
                label = 'None'
            self.img_names.append(img_name)
            self.images.append(img)
            self.labels.append(label)

        self.transform = transform_test

        with open('/home/duzilin/Code/Event/BLIP/Unify/caption/'+Caption+'_m2e2.json', 'r') as f:
            self.caption = json.load(f)

        self.doc2sens = {}
        text_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_only_event.json').read())
        text_multimedia_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_multimedia_event.json').read())
        text_event.extend(text_multimedia_event)
        for sen in text_event:
            voa_name = sen['sentence_id'][0:sen['sentence_id'].rfind('_')]
            self.doc2sens.setdefault(voa_name, [])
            self.doc2sens[voa_name].append(sen['sentence'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transform(image)
        label = LABELS_list.index(self.labels[index])
        cap = self.caption[self.img_names[index]]['cap']

        # new attention
        voa_name = self.img_names[index][0:self.img_names[index].rfind('_')]
        sens = [cap]
        if voa_name in self.doc2sens:
            sens.extend(self.doc2sens[voa_name])
        for i in range(5):
            sens.append(cap)
        cap = sens[:5]

        return image, label, cap



class VOA_Dataset(Dataset):
    def __init__(self, ann_file):
        print(f'Loading csv data from {ann_file}.')
        df = pd.read_csv(ann_file).dropna().astype(str)
        df = filter(df)

        self.images = df['filepath'].tolist()
        self.captions = df['title'].tolist()
        self.ann_pretrain = []
        for index, image in enumerate(self.images):
            ann={'image':image,'caption':self.captions[index]}
            self.ann_pretrain.append(ann)
        self.annotation = self.ann_pretrain
        self.transform = transform_test

        print('Done loading data.')


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        ann['image'] = ann['image'].split('zilin/')[1]
        image = Image.open(ann['image']).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(ann['caption'], 100)

        return image, caption

def filter(df):
    flag = 0
    images = []
    captions = []
    for index,row in df.iterrows():
        image = row['filepath']
        caption = row['title']

        if (len(caption.split(' '))<6):
            flag += 1
            continue
        if not (u'\u0041' <= caption[0] <= u'\u005a') or (u'\u0061' <= caption[0] <= u'\u007a'):
            if not u'\u0030' <= caption[0] <= u'\u0039':
                if not (u'\u0041' <= caption[1] <= u'\u005a') or (u'\u0061' <= caption[1] <= u'\u007a'):
                    if not u'\u0030' <= caption[1] <= u'\u0039':
                        flag += 1
                        continue

        images.append(image)
        captions.append(caption)

    data ={'filepath':images,'title':captions}
    return pd.DataFrame(data)