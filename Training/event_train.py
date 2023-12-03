
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup
import wandb
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig, CLIPTokenizerFast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from collections import OrderedDict
from event_dataset import imSitu_dataset, ACE_Dataset, m2e2_img_Dataset, sr_to_ace_mapping, LABELS_list, star_verb_list, verb_list, max_val, min_val, eps_size
import pickle
from event_model import Unify_model_new
from tqdm.contrib import tzip
import copy
# evaluate_conll_file and refine can be referred to https://github.com/jianliu-ml/Multimedia-EE/blob/main/code/textualEE/refine_result.py
from conlleval import evaluate_conll_file
from refine_result import refine
from torch.autograd import Variable
from utils import m2e2_collate, my_collate, mixup_data, get_m2e2_logits

import sys
sys.path.append("..")
import MEE.textualEE.config
idx2tag = MEE.textualEE.config.idx2tag

CKPT = ''
CKPT_OUTPUT = 'output/general/git/decouple'
Event_Output_Path = CKPT_OUTPUT + '/event'
Path(Event_Output_Path).mkdir(parents=True, exist_ok=True)
Tag = ''


def create_train_dataset(config_para):

    split = config_para.T_split
    imSitu_train_bs = config_para.V_train_bs
    ace_train_bs = config_para.T_train_bs
    val_bs = config_para.val_bs
    vote = config_para.V_vote

    train_dataset_imSitu = imSitu_dataset('train', vote)
    val_dataset_imSitu = imSitu_dataset('val', vote)

    # 14670+873+711
    with open("Code/Event/BLIP/MEE/textualEE/data/ace_examples_new.pkl", "rb") as f:
        ace = pickle.load(f)
    train = ace[:split]
    val = ace[split:]

    train_dataset_ace = ACE_Dataset(train)
    val_dataset_ace = ACE_Dataset(val)

    train_params_imSitu = {
        'batch_size': imSitu_train_bs,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    }
    train_params_ace = {
        'batch_size': ace_train_bs,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'collate_fn': my_collate
    }
    val_params_imSitu = {
        'batch_size': val_bs,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True
    }
    val_params_ace = {
        'batch_size': val_bs,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'collate_fn': my_collate
    }

    train_dataloader_ace = DataLoader(train_dataset_ace, **train_params_ace)
    val_dataloader_ace = DataLoader(val_dataset_ace, **val_params_ace)
    train_dataloader_imSitu = DataLoader(train_dataset_imSitu, **train_params_imSitu)
    val_dataloader_imSitu = DataLoader(val_dataset_imSitu, **val_params_imSitu)

    return train_dataloader_ace,val_dataloader_ace,train_dataloader_imSitu,val_dataloader_imSitu

def create_test_dataset(val_bs):

    with open("Code/Event/BLIP/MEE/textualEE/data/m2e2_examples_fullimg.pkl", "rb") as f:
        m2e2 = pickle.load(f)
    m2e2_dataset_text = ACE_Dataset(m2e2)
    val_params_txt = {
        'batch_size': val_bs,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
        'collate_fn': my_collate
    }
    val_params_img = {
        'batch_size': val_bs,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
        'collate_fn': m2e2_collate
    }
    m2e2_dataloader_text = DataLoader(m2e2_dataset_text, **val_params_txt)
    m2e2_dataset_img = m2e2_img_Dataset()
    m2e2_dataloader_img = DataLoader(m2e2_dataset_img, **val_params_img)
    return m2e2_dataloader_text, m2e2_dataloader_img, m2e2_dataset_img.images

@torch.no_grad()
def get_m2e2_logits_vote(config_para, epo, model, data_loader, device, M2E2, m2e2_img_list=None, topk=10):

    pred_list = []
    ground_list = []
    logits_list = []

    for images, labels, sentences in tqdm(data_loader):

        loss,logits = model.forward2((images, labels, sentences), device, M2E2)
        logits_list.append(logits)

        value_list = torch.topk(logits, topk, dim=-1, largest=True)[0].cpu()
        index_list = torch.topk(logits, topk, dim=-1, largest=True)[1].cpu()

        for index, value, label in zip(index_list, value_list, labels):
            ground = LABELS_list[label]
            ground_list.append(ground)
            pred_list.append(ground)

    Label = {}
    logits_list = torch.cat(logits_list, dim=0)
    for p, g, img, logit in zip(pred_list, ground_list, m2e2_img_list,logits_list):
        img_name = img.split('/')[-1][:-4]
        Label[img_name] =  (p, g, logit)

    with open(Event_Output_Path + "/before_refine.pkl", "wb") as f:
        pickle.dump(Label, f)

def _transfer(l, p, idx2tag):
    if l == 'O':
        l = 'O'
    else:
        l = 'B-' + l.replace('-', '_')
    p = idx2tag[p]
    if p == 'O':
        p = 'O'
    else:
        p = 'B-' + p.replace('-', '_')
    return l, p

@torch.no_grad()
def evaluate_text(config_para, epo, model, dataloader, device, M2E2):
    if M2E2==True:
        filename = Event_Output_Path + '/predict_m2e2_{}{}.conll'.format(epo, Tag)
    else:
        filename = Event_Output_Path + '/predict_val_{}{}.conll'.format(epo, Tag)
    fileout = open(filename, 'w')
    all_words = []
    all_labels = []
    all_predicts = []
    total_loss = 0
    for batch in tqdm(dataloader):
        predicts, loss = model.predict_text_label(batch, device)
        total_loss += loss.item()
        all_predicts.extend(predicts)
        words, labels = batch[-3], batch[-2]
        all_words.extend(words)
        all_labels.extend(labels)
        # break

    for words, labels, predicts in zip(all_words, all_labels, all_predicts):
        for w, l, p in zip(words, labels, predicts):
            l, p = _transfer(l, p, idx2tag)
            print(w, l, p, file=fileout)
        print(file=fileout)
    fileout.close()

    print("The results before refined: ")
    with open(filename) as fout:
        eval_results = evaluate_conll_file(fout)

    if M2E2 == True:
        print("The results after refined: ")
        refined_filename = Event_Output_Path + '/predict_M2E2_{}_refined{}.conll'.format(epo, Tag)
        eval_results, sen2type_dict = refine(filename, refined_filename)
        wandb.log({"M2E2 Text Loss": total_loss})
        wandb.log({"M2E2 Text Precision": eval_results[0]})
        wandb.log({"M2E2 Text Recall": eval_results[1]})
        wandb.log({"M2E2 Text F1": eval_results[2]})
        if config_para.save:
            with open(os.path.join(Event_Output_Path + "/TEE_{}.json".format(epo)), 'w') as f:
                json.dump(sen2type_dict, f)
    else:
        wandb.log({"ACE Val Loss": total_loss})
        wandb.log({"ACE Val Precision": eval_results[0]})
        wandb.log({"ACE Val Recall": eval_results[1]})
        wandb.log({"ACE Val F1": eval_results[2]})

    return total_loss

@torch.no_grad()
def evaluate_img(config_para, epo, model, data_loader, device, M2E2, m2e2_img_list=None):
    acc_total = 0
    p_total = 0
    g_total = 0
    acc_event_total = 0
    total = 0
    loss = 0

    pred_list = []
    ground_list = []

    for images, labels, sentences in tqdm(data_loader):
        total += len(labels)
        images = images
        labels = labels
        tmp_acc, tmp_p, tmp_g, tmp_loss, tmp_acc_event, pred, ground, _ = model.predict_img_label((images, labels, sentences),device,M2E2)
        acc_total += tmp_acc
        p_total += tmp_p
        g_total += tmp_g
        acc_event_total += tmp_acc_event
        loss += tmp_loss.item() * len(labels)

        pred_list.extend(pred)
        ground_list.extend(ground)
        # break

    if p_total == 0:
        p = 0
    else:
        p = acc_event_total / p_total
    r = acc_event_total / g_total
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    acc = acc_total / total
    loss = loss / total

    if M2E2 == True:
        wandb.log({"M2E2 Img Precision": p,
                   "M2E2 Img Recall": r,
                   "M2E2 Img F1": f1,
                   "M2E2 Img loss": loss,
                   "Epoch": epo},commit=False)
    else:
        wandb.log({"Img Val Precision": p,
                   "Img Val Recall": r,
                   "Img Val F1": f1,
                   "Img Val Acc": acc,
                   "imSitu Val Loss": loss},commit=False)

    if M2E2 == True and config_para.save:
        Label = {}
        for p, g, img in zip(pred_list, ground_list, m2e2_img_list):
            img_name = img.split('/')[-1][:-4]
            Label[img_name] = (p, g)
        with open(os.path.join(Event_Output_Path, "VEE_{}.json".format(epo)), 'w') as f:
            json.dump(Label, f)
    return loss

@torch.no_grad()
def evaluate_img_vote(config_para, epo, model, data_loader, device, M2E2, m2e2_img_list=None, topk=10):

    total = 0
    loss_total = 0.0
    acc_total = 0
    p_total = 0
    g_total = 0

    pred_list = []
    ground_list = []

    for images, labels, sentences in tqdm(data_loader):

        loss,logits = model.forward2((images, labels, sentences), device, M2E2)
        total += len(labels)
        logits = F.softmax(logits,-1)

        if M2E2:

            value_list = torch.topk(logits, topk, dim=-1, largest=True)[0].cpu()
            index_list = torch.topk(logits, topk, dim=-1, largest=True)[1].cpu()

            for index, value, label in zip(index_list, value_list, labels):
                score = torch.zeros(len(LABELS_list))
                for index_item, value_item in zip(index, value):
                    verb = verb_list[index_item]
                    if verb in star_verb_list:
                        ace_type = LABELS_list.index(sr_to_ace_mapping[verb])
                        score[ace_type] += value_item
                    else:
                        score[0] += value_item

                pred = LABELS_list[torch.argmax(score, dim=-1).item()]
                ground = LABELS_list[label]

                if pred !="None":
                    p_total+=1
                    if pred==ground:
                        acc_total+=1
                if ground !="None":
                    g_total+=1

                pred_list.append(pred)
                ground_list.append(ground)

        else:
            loss_total += loss.item() * len(labels)
            classifications = torch.argmax(logits, -1)
            classifications = list(classifications.cpu().numpy())
            for p,g in zip(classifications, labels):
                if p==g:
                    acc_total+=1
        # break

    if M2E2 == True:
        if p_total == 0:
            p = 0
        else:
            p = acc_total / p_total
        r = acc_total / g_total
        if p + r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
        wandb.log({"M2E2 Img Precision top{}".format(topk): p,
                   "M2E2 Img Recall top{}".format(topk): r,
                   "M2E2 Img F1 top{}".format(topk): f1},commit=False)

    else:
        loss = loss_total / total
        acc = acc_total / total
        wandb.log({"Img Val Acc": acc,
                   "imSitu Val Loss": loss},commit=False)

    if M2E2 == True and config_para.save:
        Label = {}
        for p, g, img in zip(pred_list, ground_list, m2e2_img_list):
            img_name = img.split('/')[-1][:-4]
            Label[img_name] = (p, g)
        with open(os.path.join(Event_Output_Path, "VEE_{}.json".format(epo)), 'w') as f:
            json.dump(Label, f)


#### Training-imSitu ####
def train_img(config_para, model,train_dataloader_imSitu, val_dataloader_imSitu,m2e2_dataloader_img, m2e2_img_list, device):
    max_val_loss = 999

    print("Learnable Parameters: ")
    if config_para.decouple:
        for name, p in model.named_parameters():
            if 'clip_fc' in name:
                print(name)
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        if config_para.round1:
            for name, p in model.named_parameters():
                # if 'clip' in name:
                if 'clip' in name and 'norm' in name:
                    print(name)
                    p.requires_grad = True
                elif 'meta_net' in name or 'V_sim_proj' in name or 'attention_block' in name:
                    print(name)
                    p.requires_grad = True
                elif 'clip_fc' in name:
                    print(name)
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for name, p in model.named_parameters():
                if 'attention_block' in name or 'clip_fc' in name:
                    print(name)
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    V_num_warmup_steps = int(config_para.V_warmup_epochs * len(train_dataloader_imSitu))
    V_num_training_steps = config_para.V_max_epoch * len(train_dataloader_imSitu)
    V_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    V_optimizer = AdamW(V_parameters, lr=config_para.V_lr, correct_bias=False, weight_decay=config_para.weight_decay)
    V_scheduler = get_cosine_schedule_with_warmup(V_optimizer, num_warmup_steps=V_num_warmup_steps,
                                                  num_training_steps=V_num_training_steps)  # PyTorch scheduler

    for epo in range(config_para.V_max_epoch):
        print(f"\nTraining epoch in VEE: {epo}")

        if config_para.decouple:
            train_dataloader_ace, val_dataloader_ace, train_dataloader_imSitu, val_dataloader_imSitu = create_train_dataset(config_para)

        model.train()
        flag = 0
        for imSitu_batch in tqdm(train_dataloader_imSitu):

            loss = model('imSitu', imSitu_batch, device)
            if flag % 5 == 0:
                wandb.log({"imSitu Training Loss": loss.item(),
                           "VEE Lr": V_optimizer.param_groups[0]["lr"]})

            loss.backward()
            V_optimizer.step()
            V_scheduler.step()
            model.zero_grad()
            flag += 1
            # break

        model.eval()
        print("Evaluate the img datasets")
        if config_para.V_vote:
            evaluate_img_vote(config_para, epo, model, val_dataloader_imSitu, device, M2E2=False)
            evaluate_img_vote(config_para, epo, model, m2e2_dataloader_img, device, M2E2=True,
                              m2e2_img_list=m2e2_img_list, topk=1)
            evaluate_img_vote(config_para, epo, model, m2e2_dataloader_img, device, M2E2=True,
                              m2e2_img_list=m2e2_img_list, topk=3)
            evaluate_img_vote(config_para, epo, model, m2e2_dataloader_img, device, M2E2=True,
                              m2e2_img_list=m2e2_img_list, topk=5)
            evaluate_img_vote(config_para, epo, model, m2e2_dataloader_img, device, M2E2=True,
                              m2e2_img_list=m2e2_img_list, topk=10)
        else:
            val_loss = evaluate_img(config_para, epo, model, val_dataloader_imSitu, device, M2E2=False)
            evaluate_img(config_para, epo, model, m2e2_dataloader_img, device, M2E2=True, m2e2_img_list=m2e2_img_list)
        if config_para.save:
            save_path = config_para.output_dir + '/model_V_{}.pth'.format(epo)
            torch.save(model.state_dict(), save_path)

        if max_val_loss>val_loss:
            max_val_loss = val_loss
            max_epo = epo
        # break

    return max_epo


#### Training-ACE2005 ####
def train_txt(config_para, model,train_dataloader_ace, val_dataloader_ace, m2e2_dataloader_text, T_max_epoch, device, T_optimizer=None, T_scheduler=None):
    if T_optimizer is None:
        print("Learnable Parameters: ")
        if config_para.decouple:
            for name, p in model.named_parameters():
                if 'bert_fc' in name:
                    print(name)
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            if config_para.round1:
                for name, p in model.named_parameters():
                    if 'T_sim_proj' in name or 'bert_fc' in name:
                        print(name)
                        p.requires_grad = True
                    # elif 'bert' in name and 'LayerNorm' in name:
                    elif 'bert' in name:
                        print(name)
                        p.requires_grad = True
                    elif 'attention_block' in name:
                        print(name)
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                for name, p in model.named_parameters():
                    if 'attention_block' in name or 'bert_fc' in name:
                        print(name)
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        T_num_warmup_steps = int(config_para.T_warmup_epochs * len(train_dataloader_ace))
        T_num_training_steps = config_para.T_max_epoch * len(train_dataloader_ace)
        T_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        T_optimizer = AdamW(T_parameters, lr=config_para.T_lr, correct_bias=False,
                            weight_decay=config_para.weight_decay)
        T_scheduler = get_cosine_schedule_with_warmup(T_optimizer, num_warmup_steps=T_num_warmup_steps,
                                                      num_training_steps=T_num_training_steps)
    else:
        print("Resume T_optimizer and T_scheduler")

    for epo in range(T_max_epoch):
        print(f"\nTraining epoch in TEE: {epo}")
        model.train()
        flag = 0
        for ace_batch in tqdm(train_dataloader_ace):
            loss = model('ace', ace_batch, device)
            if flag % 5 == 0:
                wandb.log({"ACE Training Loss": loss.item(),
                           "TEE Lr": T_optimizer.param_groups[0]["lr"]})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            T_optimizer.step()
            T_scheduler.step()
            model.zero_grad()
            flag += 1
            # break

        model.eval()
        print("Evaluate the txt datasets")
        loss_val_text = evaluate_text(config_para, epo, model, val_dataloader_ace, device, M2E2=False)
        evaluate_text(config_para, epo, model, m2e2_dataloader_text, device, M2E2=True)
        if config_para.save:
            save_path = config_para.output_dir + '/model_T_{}.pth'.format(epo)
            torch.save(model.state_dict(), save_path)
        # break




def main(args):

    wandb.init(project="Unifying Model",
               # name="decouple_git",
               )
    #### Hyperparameters ####
    config_para = wandb.config  # Initialize config_para
    config_para.weight_decay = 0.05
    config_para.seed = 1
    config_para.val_bs = 128
    config_para.output_dir = CKPT_OUTPUT
    config_para.event_path = Event_Output_Path
    config_para.save = True

    # In the first round:
    #     config_para.decouple = False
    #     config_para.round1 = True
    # In the second round:
    #     config_para.decouple = False
    #     config_para.round1 = False
    # In the third round:
    #     config_para.decouple = True
    #     config_para.round1 = False
    config_para.decouple = False
    config_para.round1 = False

    config_para.CKPT = CKPT

    config_para.T_lr = 1e-5
    config_para.T_max_epoch = 5
    config_para.T_warmup_epochs = 0.0
    config_para.T_train_bs = 10
    config_para.T_split = 15543
    config_para.T_epoch1 = 5

    config_para.V_lr = 1e-5
    config_para.V_max_epoch = 10
    config_para.V_warmup_epochs = 0.0
    config_para.V_train_bs = 64
    config_para.V_epoch1 = 10
    config_para.V_vote = False
    config_para.V_LS = 0.0
    config_para.V_Mixup = 0.0

    config_para.text = 'blip'
    config_para.img = 'dif2-1'

    config_para.resize = 0.7
    config_para.aug_n = 2
    config_para.aug_m = 5

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = config_para.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)  # Current GPU
    cudnn.deterministic = True  # Close optimization


    #### Dataset ####
    print("Creating dataset")
    train_dataloader_ace, val_dataloader_ace, train_dataloader_imSitu, val_dataloader_imSitu = create_train_dataset(config_para)
    m2e2_dataloader_text, m2e2_dataloader_img, m2e2_img_list = create_test_dataset(config_para.val_bs)
    print("ACE: {} steps in one epoch".format(len(train_dataloader_ace)))
    print("imSitu: {} steps in one epoch".format(len(train_dataloader_imSitu)))

    #### Training ####
    model = Unify_model_new("bert-base-cased", '', config_para)
    if len(CKPT)>1:
        model.load_state_dict(torch.load(CKPT))
        print('resume bert Unify_model from %s' % CKPT)
    model.to(device)

    max_epo = train_img(config_para, model, train_dataloader_imSitu, val_dataloader_imSitu, m2e2_dataloader_img, m2e2_img_list, device)
    model.load_state_dict(torch.load(config_para.output_dir + '/model_V_{}.pth'.format(max_epo)))
    print('resume bert Unify_model from %s' % CKPT)
    if not config_para.decouple:
        train_txt(config_para, model, train_dataloader_ace, val_dataloader_ace, m2e2_dataloader_text, config_para.T_epoch1, device)

    print("Evaluate the img datasets")
    evaluate_img(config_para, config_para.V_max_epoch, model, val_dataloader_imSitu, device, M2E2=False)
    evaluate_img(config_para, config_para.V_max_epoch, model, m2e2_dataloader_img, device, M2E2=True, m2e2_img_list=m2e2_img_list)

    print("Evaluate the txt datasets")
    evaluate_text(config_para, 99, model, val_dataloader_ace, device, M2E2=False)
    evaluate_text(config_para, 99, model, m2e2_dataloader_text, device, M2E2=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)











