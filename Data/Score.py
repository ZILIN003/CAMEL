import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pickle
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig, CLIPTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json


class m2e2_dataset(Dataset):
    def __init__(self, transform,data_path,image_path):
        print(f'Loading data of M2E2.')
        self.data={}

        text_only_event = json.loads(open(data_path+'text_only_event.json').read())
        text_multimedia_event = json.loads(open(data_path+'text_multimedia_event.json').read())
        image_only_event = json.loads(open(data_path+'image_only_event.json').read())
        image_multimedia_event = json.loads(open(data_path+'image_multimedia_event.json').read())
        crossmedia_coref = open(data_path+'crossmedia_coref.txt', 'r').readlines()
        self.image_path = image_path

        for to in text_only_event:
            voa_name = to['sentence_id'][0:to['sentence_id'].rfind('_')]
            self.data.setdefault(voa_name,{})
            self.data[voa_name].setdefault('article',{})
            self.data[voa_name]['article'].update({to['sentence_id']:to})
        for tm in text_multimedia_event:
            voa_name = tm['sentence_id'][0:tm['sentence_id'].rfind('_')]
            self.data.setdefault(voa_name, {})
            self.data[voa_name].setdefault('article',{})
            if tm['sentence_id'] in self.data[voa_name]['article']:
                self.data[voa_name]['article'][tm['sentence_id']]['golden-event-mentions'].extend(tm)
            else:
                self.data[voa_name]['article'].update({tm['sentence_id']: tm})
            # self.data[voa_name]['article'].update({tm['sentence_id']:tm})

        for image_name,io in image_only_event.items():
            voa_name = image_name[0:image_name.rfind('_')]
            # skip the documents which only contains images（234-229 = 5）
            # skip 10 images containing VEE (391 - 381 = 10)
            if voa_name not in self.data:
                continue
            self.data.setdefault(voa_name, {})
            self.data[voa_name].setdefault('images', {})
            self.data[voa_name]['images'].update({image_name:io})
        for image_name,im in image_multimedia_event.items():
            voa_name = image_name[0:image_name.rfind('_')]
            self.data[voa_name].setdefault('images', {})
            self.data[voa_name]['images'].update({image_name:im})

        # Total: 309, Img:203, Text:192
        for row in crossmedia_coref:
            tmp_list = row.split('\t')
            img_path = tmp_list[1]
            text_path = tmp_list[0]
            voa_name = text_path[0:text_path.rfind('_')]
            self.data[voa_name].setdefault('coref', {})
            self.data[voa_name]['coref'] = (text_path,img_path,tmp_list[2].split('\n')[0])

        self.voa_file=list(self.data.keys())
        self.transform = transform
        print('Done loading data.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        voa_file = self.voa_file[index]
        document = self.data[voa_file]
        doc_sent=[]
        doc_img=[]
        sen_id_list = []
        img_id_list = []
        flag=False
        for sen_id,sen in document['article'].items():
            doc_sent.append(sen['sentence'])
            sen_id_list.append(sen_id)
            if flag:
                continue
            if len(sen['image'])>0:
                flag=True
                for image_name in sen['image']:
                    path = self.image_path+image_name
                    if os.path.exists(path):
                        img_id_list.append(image_name)
                        image = Image.open(path).convert('RGB')
                        image = self.transform(image)
                        doc_img.append(image.unsqueeze(0))

        return voa_file,document,doc_sent,sen_id_list,doc_img,img_id_list


def create_dataset(dataset, config, min_scale=0.5, ROOT=''):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = m2e2_dataset(data_path=ROOT + 'Datasets/M2E2/voa/m2e2_annotations/',
                           image_path=ROOT + 'Datasets/M2E2/voa/m2e2_rawdata/image/image/',
                           transform=transform_test)
    return dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, drop_last=False):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = drop_last
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


MODEL_TYPE = "openai/clip-vit-base-patch16"
print("Loading model")
model = CLIPModel.from_pretrained(MODEL_TYPE)
model = model.to(device)
model.eval()
tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_TYPE)

print("Creating dataset")
dataset = create_dataset('m2e2', {'image_size': 224})
print('number of training samples: %d' % len(dataset))
samplers = create_sampler([dataset], [False], 1, 0)
data_loader = \
create_loader([dataset], samplers, batch_size=[1], num_workers=[2], is_trains=[False], collate_fns=[None])[0]
score_dict = {}

for voa_file, document, doc_sent, sen_id_list, doc_img, img_id_list in tqdm(data_loader):
    sentence = []
    sen_id = []
    img_id = []

    for sen, i in zip(doc_sent, sen_id_list):
        sentence.append(sen[0])
        sen_id.append(i[0])

    for img in img_id_list:
        img_id.append(img[0])

    if len(doc_img) == 0:
        continue

    doc_img = torch.cat(doc_img, 0).reshape(-1, 3, 224, 224).to(device, non_blocking=True)

    with torch.no_grad():
        text = tokenizer(text=sentence, padding=True, max_length=77, truncation=True, return_tensors="pt").to(device)
        outputs = model(**text, pixel_values=doc_img)
        score = outputs.logits_per_image

    score_dict.update({voa_file[0]: {'sen_id_list': sen_id, 'score': score, 'img_id_list': img_id}})

with open("CLIP_Score.pkl", "wb") as f:
    pickle.dump(score_dict, f)