from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# from pycocotools.coco import COCO
from utils import swig_box_xyxy_to_cxcywh, nested_tensor_from_tensor_list, get_anchor
from copy import deepcopy

import skimage.io
import skimage.transform
import skimage.color
import skimage
import json
import cv2
import copy
from PIL import Image
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')

Caption = 'blip'

LABELS_list= ['Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']
ROLES_list = ['Money', 'Target', 'Victim', 'Instrument', 'Agent', 'Artifact', 'Entity', 'Giver', 'Recipient', 'Attacker', 'Vehicle', 'Place', 'Person']
type2role = {'Conflict:Attack': ['Target', 'Attacker', 'Instrument'],
            'Conflict:Demonstrate': ['Entity', 'Place'], # 'Instrument','Police'
            'Contact:Meet': ['Entity'],
            'Contact:Phone-Write': ['Entity', 'Instrument'],
            'Justice:Arrest-Jail': ['Person', 'Agent', 'Instrument'],
            'Life:Die': ['Victim', 'Instrument'],
            'Movement:Transport': ['Agent', 'Artifact', 'Vehicle'],
            'Transaction:Transfer-Money': ['Money', 'Recipient', 'Giver']}

vidx_ridx=[]
for lable in LABELS_list:
    temp_list = []
    for role in type2role[lable]:
        temp_list.append(ROLES_list.index(role))
    vidx_ridx.append(temp_list)

verb2ace = {}
verb2ace_arg = {}
verb2ace_arg_new = {}
for line in open('Datasets/M2E2/imSitu/ace_sr_mapping.txt'):
    fields = line.strip().split()
    key = LABELS_list.index(fields[2].replace('||', ':').replace('|', '-'))

    verb2ace[fields[0]] = key
    if fields[3] in type2role[fields[2].replace('||', ':').replace('|', '-')]:
        verb2ace_arg.setdefault(fields[0],{})
        verb2ace_arg[fields[0]].setdefault(fields[3],[])
        verb2ace_arg[fields[0]][fields[3]].append(fields[1])

        verb2ace_arg_new.setdefault(fields[0],{})
        verb2ace_arg_new[fields[0]][fields[1]] = fields[3]


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, is_training, transform=None):

        self.transform = transform
        self.is_training = is_training
        self.color_change = transforms.Compose([transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.05), transforms.RandomGrayscale(p=0.3)])

        if self.is_training:

            SWiG_json = json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "train.json"), 'r'))
            SWiG_json.update(json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "dev.json"), 'r')))
            self.caption = json.load(open('Code/Event/BLIP/Unify/caption/'+Caption+'_train.json', 'r'))


            self.image_data = self._read_annotations(SWiG_json)
            self.image_names = list(self.image_data.keys())

        else:

            SWiG_json = json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "test.json"), 'r'))
            self.image_data = self._read_annotations(SWiG_json)
            self.image_names = list(self.image_data.keys())
            with open('Code/Event/BLIP/Unify/caption/'+Caption+'_val.json', 'r') as f:
                self.caption = json.load(f)


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):

        img = self.load_image(idx)
        verb = self.image_names[idx].split('_')[0]
        verb_idx = verb2ace[verb]
        cap = self.caption[self.image_names[idx]]['cap']

        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx}
        if self.transform:
            sample = self.transform(sample)

        sample['sentence'] = cap

        return sample

    def load_image(self, image_index):

        im = Image.open(os.path.join('Datasets/M2E2/imSitu/images_512/', self.image_names[image_index]))
        im = im.convert('RGB')

        if self.is_training:
            im = np.array(self.color_change(im))
        else:
            im = np.array(im)

        return im.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))* -1

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            annotation = np.zeros((1, 5))* -1  # allow for 3 annotations

            annotation[0, 0] = a['x1']
            annotation[0, 1] = a['y1']
            annotation[0, 2] = a['x2']
            annotation[0, 3] = a['y2']

            annotation[0, 4] = a['role']
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, json):
        result = {}

        num1=0
        num2=0

        for image in json:
            total_anns = 0
            verb = json[image]['verb']
            if verb not in verb2ace:
                continue
            num1+=1
            result[image] = []
            ace_roles = vidx_ridx[verb2ace[verb]]

            for ace_role in ace_roles:
                flag = True

                if ROLES_list[ace_role] in verb2ace_arg[verb]:
                    for role in verb2ace_arg[verb][ROLES_list[ace_role]]:
                        if json[image]['bb'][role][0]!=-1:
                            flag = False
                            total_anns += 1
                            [x1, y1, x2, y2] = json[image]['bb'][role]
                            result[image].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'role': ace_role})
                            num2 += 1
                            break
                    if flag:
                        total_anns += 1
                        [x1, y1, x2, y2] = [-1,-1,-1,-1]
                        result[image].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'role': ace_role})
                else:
                    total_anns += 1
                    [x1, y1, x2, y2] = [-1, -1, -1, -1]
                    result[image].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'role': ace_role})

            while total_anns < 3:
                total_anns += 1
                [x1, y1, x2, y2] = [-1, -1, -1, -1]
                result[image].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'role': -1})

        print("{} images and {} roles in dataset".format(num1,num2))

        return result



def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    shift_0 = [s['shift_0'] for s in data]
    shift_1 = [s['shift_1'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)
    verb_role_indices = [vidx_ridx[s['verb_idx']] for s in data]
    verb_role_indices = [torch.tensor(vri) for vri in verb_role_indices]

    heights = [s['height'] for s in data]
    widths = [s['width'] for s in data]
    sentences = [s['sentence'] for s in data]

    ground_truth = [s['ground_truth'] for s in data]

    batch_size = len(imgs)
    max_height = 224
    max_width = 224

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, shift_0[i]:shift_0[i] + img.shape[0], shift_1[i]:shift_1[i] + img.shape[1], :] = img
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    widths = torch.tensor(widths).float()
    heights = torch.tensor(heights).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scales = torch.tensor(scales).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()


    return ((padded_imgs,sentences),
            [{'verbs': vi,
              'roles': vri,
              'boxes': swig_box_xyxy_to_cxcywh(annot[:, :4], mw, mh, gt=True),
              'width': w,
              'height': h,
              'shift_0': s0,
              'shift_1': s1,
              'scale': sc,
              'max_width': mw,
              'max_height': mh,
              'img_name': im,
              'ground_truth':gt}
             for vi, vri, annot, w, h, s0, s1, sc, im, gt in zip(verb_indices, verb_role_indices, annot_padded, widths, heights, shift_0, shift_1, scales, img_names, ground_truth)])


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_for_training):
        self.is_for_training = is_for_training


    def __call__(self, sample, min_side=224, max_side=224):
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        ground_truth = deepcopy(torch.from_numpy(sample['annot']))

        rows_orig, cols_orig, cns_orig = image.shape
        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # !!!!!
        # if self.is_for_training:
        #     scale_factor = random.choice([1, 0.75, 0.5])
        #     scale = scale*scale_factor

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
        rows, cols, cns = image.shape

        new_image = np.zeros((rows, cols, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        shift_1 = int((224 - cols) * .5)
        shift_0 = int((224 - rows) * .5)

        annots[:, :4][annots[:, :4] > 0] *= scale

        annots[:, 0][annots[:, 0] > 0] = annots[:, 0][annots[:, 0] > 0] + shift_1
        annots[:, 1][annots[:, 1] > 0] = annots[:, 1][annots[:, 1] > 0] + shift_0
        annots[:, 2][annots[:, 2] > 0] = annots[:, 2][annots[:, 2] > 0] + shift_1
        annots[:, 3][annots[:, 3] > 0] = annots[:, 3][annots[:, 3] > 0] + shift_0

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx'],
                'shift_1': shift_1, 'shift_0': shift_0, 'ground_truth':ground_truth, 'height': rows_orig, 'width':cols_orig}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):

        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0][annots[:, 0] > 0] = cols - x2[annots[:, 0] > 0]
            annots[:, 2][annots[:, 2] > 0] = cols - x_tmp[annots[:, 2] > 0]

            sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'img_name': sample['img_name'], 'verb_idx': sample['verb_idx']}



# M2E2 VAE
class M2E2Dataset(Dataset):
    """CSV dataset."""

    def __init__(self, transform=None, use_gt=False):

        self.transform = transform
        self.use_gt = use_gt

        self.labels = []
        self.images = []
        m2e2_json = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_only_event.json').read())
        m2e2_json1 = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_multimedia_event.json').read())
        m2e2_json.update(m2e2_json1)

        if not self.use_gt:
            with open("Code/Event/BLIP/Unify/output/CKPT4/att/decouple/event/VEE_0.json") as f:
                self.vee = json.load(f)

        self.image_verb_data, self.image_data = self._read_annotations(m2e2_json)
        self.image_names = list(self.image_data.keys())

        with open('Code/Event/BLIP/Unify/caption/'+Caption+'_m2e2.json', 'r') as f:
            self.caption = json.load(f)

        self.doc2sens = {}
        text_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_only_event.json').read())
        text_multimedia_event = json.loads(open(ROOT + 'Datasets/M2E2/voa/m2e2_annotations/text_multimedia_event.json').read())
        text_event.extend(text_multimedia_event)
        for sen in text_event:
            voa_name = sen['sentence_id'][0:sen['sentence_id'].rfind('_')]
            self.doc2sens.setdefault(voa_name, [])
            self.doc2sens[voa_name].append(sen['sentence'])


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):

        img = self.load_image(idx)

        sample = {'img': img, 'annot': self.image_data[self.image_names[idx]], 'img_name': self.image_names[idx], 'verb_idx': self.image_verb_data[self.image_names[idx]]}
        sample = self.transform(sample)

        # get sentences
        cap = self.caption[self.image_names[idx]]['cap']
        voa_name = self.image_names[idx][0:self.image_names[idx].rfind('_')]
        sens = [cap]
        if voa_name in self.doc2sens:
            sens.extend(self.doc2sens[voa_name])
        for i in range(5):
            sens.append(cap)
        cap = sens[:5]

        sample['sentence'] = cap

        return sample

    def load_image(self, image_index):

        im = Image.open(os.path.join('Datasets/M2E2/voa/m2e2_rawdata/image/image', self.image_names[image_index]+'.jpg'))
        im = im.convert('RGB')

        im = np.array(im)

        return im.astype(np.float32) / 255.0

    def _read_annotations(self, json):
        verb_result = {}
        arg_result = {}
        num1=0
        num2=0

        if self.use_gt:

            for image in json:
                verb = LABELS_list.index(json[image]['event_type'])
                num1+=1
                roles = json[image]['role']
                verb_result[image] = verb
                arg_result[image] = []
                for role_id in vidx_ridx[verb]:
                    temp_list = []
                    if ROLES_list[role_id] in roles:
                        for tmp in roles[ROLES_list[role_id]]:
                            temp_list.append(tmp[1:])
                        arg_result[image].append(temp_list)
                        num2 += len(roles[ROLES_list[role_id]])
                    else:
                        arg_result[image].append(temp_list)

            print("{} images and {} roles in M2E2 dataset".format(num1,num2))

            return verb_result,arg_result


        for image,ee in self.vee.items():
            if ee[0]!='None':
                num1+=1
                verb = LABELS_list.index(ee[0])
                verb_result[image] = verb
                arg_result[image] = []
                for role_id in vidx_ridx[verb]:
                    temp_list = []
                    if image in json:
                        roles = json[image]['role']
                        if ROLES_list[role_id] in roles:
                            for tmp in roles[ROLES_list[role_id]]:
                                temp_list.append(tmp[1:])
                            arg_result[image].append(temp_list)
                            num2 += len(roles[ROLES_list[role_id]])
                        else:
                            arg_result[image].append(temp_list)
                    else:
                        arg_result[image].append(temp_list)

        print("{} images and {} roles in M2E2 dataset".format(num1, num2))
        return verb_result, arg_result



def collater_m2e2(data):
    imgs = [s['img'] for s in data]
    shift_0 = [s['shift_0'] for s in data]
    shift_1 = [s['shift_1'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)
    verb_role_indices = [vidx_ridx[s['verb_idx']] for s in data]
    verb_role_indices = [torch.tensor(vri) for vri in verb_role_indices]

    heights = [s['height'] for s in data]
    widths = [s['width'] for s in data]
    sentences = [s['sentence'] for s in data]

    ground_truth = [s['ground_truth'] for s in data]

    batch_size = len(imgs)
    max_height = 224
    max_width = 224

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, shift_0[i]:shift_0[i] + img.shape[0], shift_1[i]:shift_1[i] + img.shape[1], :] = img
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    widths = torch.tensor(widths).float()
    heights = torch.tensor(heights).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scales = torch.tensor(scales).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()


    return ((padded_imgs,sentences),
            [{'verbs': vi,
              'roles': vri,
              'width': w,
              'height': h,
              'shift_0': s0,
              'shift_1': s1,
              'scale': sc,
              'max_width': mw,
              'max_height': mh,
              'img_name': im,
              'ground_truth':gt}
             for vi, vri, w, h, s0, s1, sc, im, gt in zip(verb_indices, verb_role_indices, widths, heights, shift_0, shift_1, scales, img_names, ground_truth)])

class Resizer_M2E2(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_for_training):
        self.is_for_training = is_for_training


    def __call__(self, sample, min_side=224, max_side=224):
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        ground_truth = sample['annot']

        rows_orig, cols_orig, cns_orig = image.shape
        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side


        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
        rows, cols, cns = image.shape

        new_image = np.zeros((rows, cols, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        shift_1 = int((224 - cols) * .5)
        shift_0 = int((224 - rows) * .5)


        return {'img': torch.from_numpy(new_image), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx'],
                'shift_1': shift_1, 'shift_0': shift_0, 'ground_truth':ground_truth, 'height': rows_orig, 'width':cols_orig}



class CSVDataset_detector(Dataset):
    """CSV dataset."""

    def __init__(self, is_training, transform=None):

        self.transform = transform
        self.is_training = is_training
        self.color_change = transforms.Compose([transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.05), transforms.RandomGrayscale(p=0.3)])

        if self.is_training:

            SWiG_json = json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "train.json"), 'r'))
            SWiG_json.update(json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "dev.json"), 'r')))
            self.caption = json.load(open('Code/Event/BLIP/Unify/caption/'+Caption+'_train.json', 'r'))

            self.image_data = self._read_annotations(SWiG_json)
            self.image_names = list(self.image_data.keys())

        else:

            SWiG_json = json.load(open(os.path.join('Datasets/M2E2/imSitu/SWiG_jsons/', "test.json"), 'r'))
            self.image_data = self._read_annotations(SWiG_json)
            self.image_names = list(self.image_data.keys())
            with open('Code/Event/BLIP/Unify/caption/'+Caption+'_val.json', 'r') as f:
                self.caption = json.load(f)


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):

        img = self.load_image(idx)
        verb = self.image_names[idx].split('_')[0]
        verb_idx = verb2ace[verb]
        cap = self.caption[self.image_names[idx]]['cap']
        sample = {'img': img, 'annot': self.image_data[self.image_names[idx]], 'img_name': self.image_names[idx], 'verb_idx': verb_idx}

        if self.transform:
            sample = self.transform(sample)

        sample['sentence'] = cap

        return sample

    def load_image(self, image_index):
        im = Image.open(os.path.join('Datasets/M2E2/imSitu/images_512/', self.image_names[image_index]))
        im = im.convert('RGB')
        if self.is_training:
            im = np.array(self.color_change(im))
        else:
            im = np.array(im)
        return im.astype(np.float32) / 255.0

    def _read_annotations(self, json):
        result = {}

        num1=0
        num2=0
        num3=0

        for image in json:
            verb = json[image]['verb']
            if verb not in verb2ace:
                continue

            for role,bbox in json[image]['bb'].items():
                cal_bbox = [bbox[0],bbox[1],bbox[2]-1e-5,bbox[3]-1e-5]
                if bbox[0] == -1:
                    continue
                elif role in verb2ace_arg_new[verb]:
                    result.setdefault(image,[])
                    result[image].append({'bbox': copy.deepcopy(cal_bbox), 'role': ROLES_list.index(verb2ace_arg_new[verb][role]), 'ground': copy.deepcopy(cal_bbox)})
                    num2 += 1
                elif role not in verb2ace_arg_new[verb]:
                    result.setdefault(image,[])
                    result[image].append({'bbox': copy.deepcopy(cal_bbox), 'role': 13, 'ground': copy.deepcopy(cal_bbox)})
                    num3 += 1

            if image in result:
                num1+=1
        print("{} images: {} roles {} non-roles in dataset".format(num1,num2,num3))

        return result


class Resizer_detector(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_for_training):
        self.is_for_training = is_for_training


    def __call__(self, sample, min_side=224, max_side=224):
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        rows_orig, cols_orig, cns_orig = image.shape
        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
        rows, cols, cns = image.shape

        new_image = np.zeros((rows, cols, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        shift_1 = int((224 - cols) * .5)
        shift_0 = int((224 - rows) * .5)


        for an in annots:
            an['bbox'][0] = an['bbox'][0] * scale + shift_1
            an['bbox'][1] = an['bbox'][1] * scale + shift_0
            an['bbox'][2] = an['bbox'][2] * scale + shift_1
            an['bbox'][3] = an['bbox'][3] * scale + shift_0

            an['anchor'] = get_anchor(an['bbox'], return_num=3, img_size=224, patch_size=16)

        return {'img': torch.from_numpy(new_image), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx'],
                'shift_1': shift_1, 'shift_0': shift_0, 'ground_truth':annots, 'height': rows_orig, 'width':cols_orig}

class Augmenter_detector(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, flip_x=0.5):

        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            for an in annots:

                an['bbox'][0] = cols - an['bbox'][2]
                an['bbox'][2] = cols - an['bbox'][0]

            sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx']}

        return sample

def collater_detector(data):

    imgs = [s['img'] for s in data]
    shift_0 = [s['shift_0'] for s in data]
    shift_1 = [s['shift_1'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)
    heights = [s['height'] for s in data]
    widths = [s['width'] for s in data]
    sentences = [s['sentence'] for s in data]

    ground_truth = [s['ground_truth'] for s in data]

    batch_size = len(imgs)
    max_height = 224
    max_width = 224

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, shift_0[i]:shift_0[i] + img.shape[0], shift_1[i]:shift_1[i] + img.shape[1], :] = img
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    widths = torch.tensor(widths).float()
    heights = torch.tensor(heights).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scales = torch.tensor(scales).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()


    return ((padded_imgs,sentences),
            [{'verbs': vi,
              'width': w,
              'height': h,
              'shift_0': s0,
              'shift_1': s1,
              'scale': sc,
              'max_width': mw,
              'max_height': mh,
              'img_name': im,
              'ground_truth':gt}
             for vi, w, h, s0, s1, sc, im, gt in zip(verb_indices, widths, heights, shift_0, shift_1, scales, img_names, ground_truth)])



# M2E2 VAE
class M2E2Dataset_detector(Dataset):
    """CSV dataset."""

    def __init__(self, transform=None, use_gt=False, threshold=0.75, vee_result=''):

        self.transform = transform
        self.use_gt = use_gt
        self.threshold = threshold

        self.labels = []
        self.images = []
        m2e2_json = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_only_event.json').read())
        m2e2_json1 = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_multimedia_event.json').read())
        m2e2_json.update(m2e2_json1)

        with open("Datasets/M2E2/voa/object_detection/yolov8l/bboxes.pkl", "rb") as f:
            self.detector_result = pickle.load(f)

        if not self.use_gt:
            with open(vee_result) as f:
                self.vee = json.load(f)

        self.image_verb_data, self.image_data = self._read_annotations(m2e2_json)
        self.image_names = list(self.image_data.keys())

        with open('Code/Event/BLIP/Unify/caption/'+Caption+'_m2e2.json', 'r') as f:
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
        return len(self.image_names)


    def __getitem__(self, idx):

        img = self.load_image(idx)
        sample = {'img': img, 'annot': self.image_data[self.image_names[idx]], 'img_name': self.image_names[idx], 'verb_idx': self.image_verb_data[self.image_names[idx]]}
        sample = self.transform(sample)

        # get sentences
        cap = self.caption[self.image_names[idx]]['cap']
        voa_name = self.image_names[idx][0:self.image_names[idx].rfind('_')]
        sens = [cap]
        if voa_name in self.doc2sens:
            sens.extend(self.doc2sens[voa_name])
        for i in range(5):
            sens.append(cap)
        cap = sens[:5]

        sample['sentence'] = cap

        return sample

    def load_image(self, image_index):

        im = Image.open(os.path.join('Datasets/M2E2/voa/m2e2_rawdata/image/image', self.image_names[image_index]+'.jpg'))
        im = im.convert('RGB')

        im = np.array(im)

        return im.astype(np.float32) / 255.0

    def _read_annotations(self, json):
        verb_result = {}
        arg_result = {}
        num1=0
        num2=0
        num3=0
        num4=0

        # wrong
        if self.use_gt:

            for image in json:

                detector_bboxes = self.detector_result[image+'.jpg']['xyxy']
                detector_conf = self.detector_result[image+'.jpg']['conf']
                index_selected = detector_conf > self.threshold
                bbox_candidates = detector_bboxes[index_selected].tolist()

                verb = LABELS_list.index(json[image]['event_type'])
                num1+=1
                roles = json[image]['role']
                verb_result[image] = verb
                arg_result[image] = {}
                arg_result[image]['bboxes']=bbox_candidates
                arg_result[image]['argument'] = {}
                for role_id in vidx_ridx[verb]:
                    temp_list = []
                    if ROLES_list[role_id] in roles:
                        arg_result[image]['argument'].setdefault(ROLES_list[role_id],[])
                        for tmp in roles[ROLES_list[role_id]]:
                            temp_list.append(tmp[1:])
                        arg_result[image]['argument'][ROLES_list[role_id]].append(temp_list)
                        num2 += len(roles[ROLES_list[role_id]])

            print("{} images and {} roles in M2E2 dataset".format(num1,num2))

            return verb_result,arg_result


        for image,ee in self.vee.items():
            if ee[0]!='None':
                detector_bboxes = self.detector_result[image+'.jpg']['xyxy']
                detector_conf = self.detector_result[image+'.jpg']['conf']
                index_selected = detector_conf > self.threshold
                bbox_candidates = detector_bboxes[index_selected].tolist()

                num1+=1
                verb_result[image] = LABELS_list.index(ee[0])
                arg_result[image] = {}
                arg_result[image]['bboxes'] = bbox_candidates
                arg_result[image]['argument'] = {}

                if image in json:
                    verb = LABELS_list.index(json[image]['event_type'])
                    for role_id in vidx_ridx[verb]:
                        roles = json[image]['role']
                        temp_list = []
                        if ROLES_list[role_id] in roles:
                            for tmp in roles[ROLES_list[role_id]]:
                                temp_list.append(tmp[1:])
                            num2 += len(temp_list)
                            arg_result[image]['argument'][ROLES_list[role_id]] = temp_list
            else:
                num3+=1
                if image in json:
                    verb = LABELS_list.index(json[image]['event_type'])
                    roles = json[image]['role']

                    for role_id in vidx_ridx[verb]:
                        if ROLES_list[role_id] in roles:
                            num4+= len(json[image]['role'][ROLES_list[role_id]])

        print("{} images and {} roles in M2E2 dataset".format(num1, num2))
        print("{} images and {} roles in M2E2 dataset cannot be discovered".format(num3, num4))
        return verb_result, arg_result