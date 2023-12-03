
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import json
from pathlib import Path
import sys
sys.path.append('..')
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup
import wandb
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
from V_arg_model import CoFormer,Transformer,SWiGCriterion, Classifier_VA
import sys
from V_arg_dataset import CSVDataset, Normalizer, Resizer, collater, vidx_ridx, Augmenter, M2E2Dataset, \
    collater_m2e2, Resizer_M2E2, collater_detector, CSVDataset_detector,Augmenter_detector, Resizer_detector, \
    M2E2Dataset_detector
from torchvision import transforms
sys.path.append("..")
import MEE.textualEE.config as config
idx2tag = config.idx2tag
type2role = config.type2role
idx2tag_role = config.idx2tag_role

CKPT_CLIP = "Code/Event/BLIP/CLIP/output/Finetune_CLIP16_1/checkpoint_05.pth"
CKPT = 'output/CKPT4/att/model_T_0.pth'
# CKPT = ''
vee_result = "Code/Event/BLIP/Unify/output/CKPT4/att/decouple/event/VEE_0.json"
CKPT_OUTPUT = 'output/ARG_CKPT/CKPT1'
Event_Output_Path = CKPT_OUTPUT + '/result'
Path(Event_Output_Path).mkdir(parents=True, exist_ok=True)



def create_dataset(config_para):

    dataset_train = CSVDataset(is_training=True, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(True)]))
    dataset_val = CSVDataset(is_training=False, transform=transforms.Compose([Normalizer(), Resizer(False)]))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, config_para.train_bs, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=config_para.num_workers,
                                   collate_fn=collater, batch_sampler=batch_sampler_train)
    dataloader_val = DataLoader(dataset_val, num_workers=config_para.num_workers,
                                 drop_last=False, collate_fn=collater, sampler=sampler_val)

    # M2E2
    dataset_test = M2E2Dataset(transform=transforms.Compose([Normalizer(), Resizer_M2E2(False)]), use_gt = config_para.use_gt)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, num_workers=config_para.num_workers, drop_last=False, collate_fn=collater_m2e2, sampler=sampler_test)

    return dataloader_train, dataloader_val, dataloader_test


@torch.no_grad()
def evaluate_img(model, dataloader_val, device, criterion, M2E2, config_para):
    acc_total = 0
    p_total = 0
    g_total = 0
    val_loss = 0.0
    val_loss_bbox = 0.0
    val_loss_giou = 0.0
    val_loss_conf = 0.0

    model.eval()
    with torch.no_grad():
        print("Evaluate the img datasets")
        for data, targets in tqdm(dataloader_val):
            outputs = model(data, targets, device, M2E2)
            loss_dict, result = criterion(outputs, targets, device, M2E2)

            if not M2E2:
                loss_bbox = loss_dict['loss_bbox']
                loss_giou = loss_dict['loss_giou']
                loss_conf = loss_dict['loss_bbox_conf']
                val_loss += loss_bbox+loss_giou+loss_conf
                val_loss_bbox+=loss_bbox
                val_loss_giou+=loss_giou
                val_loss_conf+=loss_conf

            p_total += result[0]
            g_total += result[1]
            acc_total += result[2]
            # if M2E2==False:
            #     break

    if not config_para.use_gt:
        g_total = 1285

    print("M2E2: {}".format(M2E2))
    print(acc_total)
    print(p_total)
    print(g_total)

    if p_total == 0:
        p = 0
    else:
        p = acc_total / p_total
    r = acc_total / g_total
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    if M2E2:
        wandb.log({"M2E2 Precision": p,
                   "M2E2 Recall": r,
                   "M2E2 F1": f1}, commit=False)
    else:
        wandb.log({"imSitu Val Loss": val_loss.item()/len(dataloader_val),
                   "imSitu Val loss_bbox": loss_dict['loss_bbox'].item()/len(dataloader_val),
                   "imSitu Val loss_giou": loss_dict['loss_giou'].item()/len(dataloader_val),
                   "imSitu Val loss_bbox_conf": loss_dict['loss_bbox_conf'].item()/len(dataloader_val),
                   "imSitu Val Precision": p,
                   "imSitu Val Recall": r,
                   "imSitu Val F1": f1}, commit=False)


#### Training-imSitu ####
def train_img(config_para, model, dataloader_train, dataloader_val, dataloader_test, criterion, device):

    for name, p in model.named_parameters():
        if 'backbone' in name:
            # if 'clip' in name and 'norm' in name:
            if 'clip' in name:
                p.requires_grad = True
            elif 'attention_block' in name and 'V_sim_proj' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

    num_warmup_steps = int(config_para.warmup_epochs * len(dataloader_train))
    num_training_steps = config_para.max_epoch * len(dataloader_train)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=config_para.lr, correct_bias=False, weight_decay=config_para.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                  num_training_steps=num_training_steps)  # PyTorch scheduler

    for epo in range(config_para.max_epoch):
        print(f"\nTraining epoch in VEE: {epo}")

        model.train()
        flag = 0
        for data, targets in tqdm(dataloader_train):

            outputs = model(data, targets, device)
            loss_dict, _ = criterion(outputs, targets, device, False)

            # return
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            if flag % 3 == 0:
                wandb.log({"imSitu Training Loss": losses.item(),
                           "imSitu Training loss_bbox": loss_dict['loss_bbox'].item(),
                           "imSitu Training loss_giou": loss_dict['loss_giou'].item(),
                           "imSitu Training loss_bbox_conf": loss_dict['loss_bbox_conf'].item(),
                           "VEE Lr": optimizer.param_groups[0]["lr"]})

            losses.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            flag += 1
            # break

        evaluate_img(model, dataloader_val, device, criterion, False, config_para)
        # evaluate_img(model, dataloader_test, device, criterion, True, config_para)



def main_new(args):

    wandb.init(project="Unifying Model Coformer")
    #### Hyperparameters ####
    config_para = wandb.config  # Initialize config_para
    config_para.weight_decay = 1e-3
    config_para.seed = 1
    config_para.output_dir = CKPT_OUTPUT
    config_para.save = True
    config_para.CKPT = CKPT
    config_para.decoder_layers = 3
    config_para.use_gt = False

    config_para.lr = 1e-5
    config_para.max_epoch = 10
    config_para.warmup_epochs = 0.0
    config_para.train_bs = 16
    config_para.num_workers = 32
    # config_para.val_bs = 16

    device = torch.device(args.device)
    seed = config_para.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)  # Current GPU
    cudnn.deterministic = True  # Close optimization


    #### Dataset ####
    print("Creating dataset")
    dataloader_train, dataloader_val, dataloader_test = create_dataset(config_para)
    print("{} steps in one epoch".format(len(dataloader_train)))

    #### Model ####
    transformer = Transformer(
        d_model=512,
        dropout=0.15,
        nhead=8,
        num_gaze_s2_dec_layers=config_para.decoder_layers,
        dim_feedforward=512
    )
    model = CoFormer(transformer, vidx_ridx=vidx_ridx, Unify_model_path = CKPT)
    criterion = SWiGCriterion()

    model.to(device)

    #### Training ####
    train_img(config_para, model, dataloader_train, dataloader_val, dataloader_test, criterion, device)

    if config_para.save:
        save_path = config_para.output_dir + '/model.pth'
        torch.save(model.state_dict(), save_path)



def create_dataset_detector(config_para):

    dataset_train = CSVDataset_detector(is_training=True, transform=transforms.Compose(
        [Normalizer(), Augmenter_detector(), Resizer_detector(True)]))
    dataset_val = CSVDataset_detector(is_training=False,
                                      transform=transforms.Compose([Normalizer(), Resizer_detector(False)]))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, config_para.train_bs, drop_last=True)
    dataloader_train = DataLoader(dataset_train, num_workers=config_para.num_workers,
                                  collate_fn=collater_detector, batch_sampler=batch_sampler_train)
    dataloader_val = DataLoader(dataset_val, num_workers=config_para.num_workers,
                                drop_last=False, collate_fn=collater_detector, sampler=sampler_val)

    # M2E2
    dataset_test = M2E2Dataset_detector(transform=transforms.Compose([Normalizer(), Resizer_M2E2(False)]),
                                        use_gt=config_para.use_gt, threshold=config_para.threshold, vee_result=vee_result)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, num_workers=config_para.num_workers, drop_last=False,
                                 collate_fn=collater_m2e2, sampler=sampler_test)

    return dataloader_train, dataloader_val, dataloader_test


@torch.no_grad()
def evaluate_img_detector(model, dataloader_val, device, M2E2):
    acc_total = 0
    p_total = 0
    g_total = 0
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        if not M2E2:
            print("Evaluate on the val datasets")
            for data, targets in tqdm(dataloader_val):
                loss, logits = model(data, targets, device)
                val_loss+=loss
                for logit, target in zip(logits, targets):
                    classifications = torch.argmax(logit, -1)
                    classifications = list(classifications.cpu().numpy())
                    for c, t in zip(classifications, target['ground_truth']):
                        if t['role'] != 13:
                            g_total+=1
                        if c!=13:
                            p_total+=1
                        if c!=13 and c == t['role']:
                            acc_total += 1
                # break
        else:
            dict_result = {}
            print("Evaluate on the M2E2 datasets")
            for data, targets in tqdm(dataloader_val):
                p_num, _, a_num = model.predict(data, targets, device, dict_result,VEE)
                acc_total += a_num
                p_total += p_num

    if M2E2:
        # visual only 627 + mm 658
        g_total = 1285

    print("M2E2: {}".format(M2E2))
    print(acc_total)
    print(p_total)
    print(g_total)

    if p_total == 0:
        p = 0
    else:
        p = acc_total / p_total
    r = acc_total / g_total
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    if M2E2:
        wandb.log({"M2E2 Precision": p,
                   "M2E2 Recall": r,
                   "M2E2 F1": f1}, commit=False)
        return dict_result
    else:
        wandb.log({"imSitu Val Loss": val_loss.item()/len(dataloader_val),
                   "imSitu Val Precision": p,
                   "imSitu Val Recall": r,
                   "imSitu Val F1": f1}, commit=False)


#### Training-imSitu ####
def train_img_detector(config_para, model, dataloader_train, dataloader_val, dataloader_test, device):

    for name, p in model.named_parameters():
        if 'backbone' in name:
            # if 'clip' in name and 'norm' in name:
            if 'clip' in name:
                p.requires_grad = True
            elif 'attention_block' in name and 'V_sim_proj' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

    num_warmup_steps = int(config_para.warmup_epochs * len(dataloader_train))
    num_training_steps = config_para.max_epoch * len(dataloader_train)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=config_para.lr, correct_bias=False, weight_decay=config_para.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                  num_training_steps=num_training_steps)  # PyTorch scheduler

    for epo in range(config_para.max_epoch):
        print(f"\nTraining epoch in VEE: {epo}")

        model.train()
        flag = 0
        for data, targets in tqdm(dataloader_train):
            loss, _ = model(data, targets, device)
            if flag % 3 == 0:
                wandb.log({"imSitu Training Loss": loss.item(),
                           "VEE Lr": optimizer.param_groups[0]["lr"]})
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            flag += 1
            # break

        evaluate_img_detector(model, dataloader_val, device, False)
        dict_result = evaluate_img_detector(model, dataloader_test, device, True)

        if config_para.save:
            with open(os.path.join(Event_Output_Path, "VAE_{}.json".format(epo)), 'w') as f:
                json.dump(dict_result, f)


def main(args):

    wandb.init(project="Unifying Model Coformer detector",
               name='combine')
    #### Hyperparameters ####
    config_para = wandb.config  # Initialize config_para
    config_para.weight_decay = 1e-3
    config_para.seed = 1
    config_para.output_dir = CKPT_OUTPUT
    config_para.save = True
    config_para.CKPT = CKPT
    config_para.use_gt = False
    config_para.threshold = 0.8
    config_para.use_cap = True

    config_para.lr = 5e-6
    config_para.max_epoch = 5
    config_para.warmup_epochs = 0.0
    config_para.train_bs = 16
    config_para.num_workers = 32
    # config_para.val_bs = 16

    device = torch.device(args.device)
    seed = config_para.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)  # Current GPU
    cudnn.deterministic = True  # Close optimization


    #### Dataset ####
    print("Creating dataset")
    dataloader_train, dataloader_val, dataloader_test = create_dataset_detector(config_para)
    print("{} steps in one epoch".format(len(dataloader_train)))

    #### Model ####
    model = Classifier_VA(Unify_model_path = CKPT, use_cap = config_para.use_cap)
    model.to(device)

    #### Training ####
    train_img_detector(config_para, model, dataloader_train, dataloader_val, dataloader_test, device)

    # if config_para.save:
    #     save_path = config_para.output_dir + '/model_VAE.pth'
    #     torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # main_new(args)
    main(args)











