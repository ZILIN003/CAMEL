
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
from T_arg_dataset import ACE_Dataset_ARG,M2E2_Dataset_ARG
import pickle
from T_arg_model import Unify_model_ARG
import sys
sys.path.append("..")
import MEE.textualEE.config as config
idx2tag = config.idx2tag
type2role = config.type2role
idx2tag_role = config.idx2tag_role

CKPT_CLIP = "Code/Event/BLIP/CLIP/output/Finetune_CLIP16_1/checkpoint_05.pth"
CKPT = 'Code/Event/BLIP/Unify/output/general/dif2/att/model_T_0.pth'

CKPT_OUTPUT = 'output/ARG_CKPT/CKPT0'
Event_Output_Path = CKPT_OUTPUT + '/result'
Path(Event_Output_Path).mkdir(parents=True, exist_ok=True)
result = 'Code/Event/BLIP/Unify/output/CKPT4/att/event/refine_best.conll'

def ace_collate(batch):

    f = torch.FloatTensor
    l = torch.LongTensor
    data_x, bert_mask, data_span, data_y, images = list(), list(), list(), list(), list()
    for data in batch:
        data_x.append(data['data_x'].tolist())
        bert_mask.append(data['bert_mask'].tolist())
        data_span.append(data['data_span'])
        data_y.append(data['data_y'])
        images.append(data['images'])

    images = torch.cat(images, 0)

    real_batch = (l(data_x),l(bert_mask).squeeze(1),data_span,data_y,f(images))

    return real_batch

def m2e2_txt_collate(data_item):

    d_x, span, ev, entity, images, sentence_id = data_item[0]['data_x'], data_item[0]['span'], data_item[0]['ev'], data_item[0]['entity'], \
                                    data_item[0]['images'], data_item[0]['sentence_id']
    trigger_res = data_item[0]['event_result']
    dict = {'data_x': d_x,
            'span': span,
            'ev': ev,
            'entity': entity,
            'images': images,
            'event_result': trigger_res,
            'sentence_id': sentence_id
            }
    return dict

def create_txt_dataset(config_para):

    split = config_para.T_split
    ace_train_bs = config_para.T_train_bs
    val_bs = config_para.val_bs

    with open("Code/Event/BLIP/MEE/textualEE/data/ace_examples_new_arg_{}.pkl".format(config_para.y_num), "rb") as f:
        ace = pickle.load(f)
    train = ace[:split]
    val = ace[split:]

    train_dataset_ace = ACE_Dataset_ARG(train)
    val_dataset_ace = ACE_Dataset_ARG(val)

    train_params_ace = {
        'batch_size': ace_train_bs,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'collate_fn': ace_collate
    }

    val_params_ace = {
        'batch_size': val_bs,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True,
        'collate_fn': ace_collate
    }

    train_dataloader_ace = DataLoader(train_dataset_ace, **train_params_ace)
    val_dataloader_ace = DataLoader(val_dataset_ace, **val_params_ace)

    with open("Code/Event/BLIP/MEE/textualEE/data/m2e2_examples_fullimg_arg.pkl", "rb") as f:
        m2e2_data = pickle.load(f)

    m2e2_dataset = M2E2_Dataset_ARG(m2e2_data,config_para.event_result)

    m2e2_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 32,
        'pin_memory': True,
        'collate_fn': m2e2_txt_collate
    }

    m2e2_txt_dataloader = DataLoader(m2e2_dataset, **m2e2_params)

    return train_dataloader_ace,val_dataloader_ace,m2e2_txt_dataloader

@torch.no_grad()
def evaluate_M2E2_text(config_para,model,m2e2_txt_dataloader,device):

    p_count, g_count, correct_count = 0, 0, 0
    dict_result ={}

    for data_item in tqdm(m2e2_txt_dataloader):

        d_x, span, ev, entity, images, sentence_id = data_item['data_x'], data_item['span'], data_item['ev'], data_item['entity'], data_item['images'], data_item['sentence_id']
        trigger_res = data_item['event_result']

        g = list()
        for e in ev:
            for a in e['arguments']:
                g.append(
                    (e['event_type'].split(':')[1], span[e['trigger']['start']][0], span[a['start']][0], a['role']))

        _, _, predicted = trigger_res
        my_span = list()
        ev_types = list()
        for idx, pred in enumerate(predicted):
            if pred != 'O':
                for e in entity:
                    # print(span[idx][0], pred, span[e['start']][0])
                    my_span.append((span[idx][0], span[e['start']][0], span[e['end']][-1]))
                    ev_types.append(pred[2:].replace('_', '-'))

        if my_span:
            f = torch.LongTensor
            d_x = f(d_x).unsqueeze(0).to('cuda')
            ms = torch.ones(d_x.size()).to('cuda')
            sp = [my_span]
            logits = model.predict_txt((d_x, ms, sp, images), device)

        p = list()
        for ms, et, logit in zip(my_span, ev_types, logits):

            value_list = torch.topk(logit,k=config_para.y_num,dim=-1,largest=True)[0]
            index_list = list(torch.topk(logit,k=config_para.y_num,dim=-1,largest=True)[1].cpu().numpy())

            for index in index_list:
                if idx2tag_role[index] in type2role[et]:
                    x = index
                    break

            if config.idx2tag_role[x] != 'O':
                l = (et, ms[0], ms[1], config.idx2tag_role[x])
                p.append(l)

        dict_result[sentence_id] = [len(set(g).intersection(set(p))),len(p)]
        g_count += len(g)
        p_count += len(p)
        correct_count += len(set(g).intersection(set(p)))

    p = correct_count / p_count
    r = correct_count / g_count
    f1 = 2 * p * r / (p + r)
    print(p, r, f1)

    wandb.log({"M2E2 Text Precision": p,
               "M2E2 Text Recall": r,
               "M2E2 Text F1":f1},commit=False)
    return dict_result

#### Training-ACE2005 ####
def train_txt(config_para, model,train_dataloader_ace, val_dataloader_ace, m2e2_txt_dataloader, device):

    print("Learnable Parameters: ")

    for name, p in model.named_parameters():
        if 'bert_fc' in name:
            print(name)
            p.requires_grad = True
        elif 'bert' in name:
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


    for epo in range(config_para.T_max_epoch):
        print(f"\nTraining epoch in TAE: {epo}")
        model.train()
        flag = 0

        for ace_batch in tqdm(train_dataloader_ace):
            loss = model('ace', ace_batch, device)
            if flag % 5 == 0:
                wandb.log({"ACE Training Loss": loss.item(),
                           "TEE Lr": T_optimizer.param_groups[0]["lr"]})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            T_optimizer.step()
            T_scheduler.step()
            model.zero_grad()
            flag += 1
            # break

        model.eval()
        # with torch.no_grad():
        #     num = 0
        #     total_loss = 0.0
        #     for ace_batch in tqdm(val_dataloader_ace):
        #         num+=len(ace_batch)
        #         loss = model('ace', ace_batch, device)
        #         total_loss += loss
        #     wandb.log({"ACE Val Loss": total_loss.item()/num},commit=False)

        with torch.no_grad():
            print("Evaluate the M2E2 txt datasets")
            dict_result = evaluate_M2E2_text(config_para, model,m2e2_txt_dataloader,device)

        if config_para.save:
            with open(os.path.join(Event_Output_Path, "TAE_{}.json".format(epo)), 'w') as f:
                json.dump(dict_result, f)



def main(args):

    wandb.init(project="Unifying Model ARG",
               name = "dif2"
               )
    #### Hyperparameters ####
    config_para = wandb.config  # Initialize config_para
    config_para.weight_decay = 0.05
    config_para.seed = 1
    config_para.val_bs = 32
    config_para.output_dir = CKPT_OUTPUT
    config_para.save = True
    config_para.ckpt = True
    config_para.y_num = 19

    config_para.T_lr = 1e-5
    config_para.T_max_epoch = 4
    config_para.T_warmup_epochs = 0.0
    config_para.T_train_bs = 2
    config_para.T_split = 3825
    config_para.event_result = result

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
    train_dataloader_ace, val_dataloader_ace, m2e2_txt_dataloader = create_txt_dataset(config_para)
    print("ACE: {} steps in one epoch".format(len(train_dataloader_ace)))

    #### Training ####
    model = Unify_model_ARG("bert-base-cased", CKPT_CLIP, config_para.y_num)
    if config_para.ckpt:
        print('resume bert Unify_model_ARG from %s' % CKPT)
        pretrained_dict = torch.load(CKPT)
        model_dict = model.state_dict()
        pretrained_dict = {key: value for key, value in pretrained_dict.items() if
                           (key in model_dict and (
                                       'attention_block' in key or 'T_sim_proj' in key or 'bert.' in key or 'clip.' in key))}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)

    train_txt(config_para, model, train_dataloader_ace, val_dataloader_ace, m2e2_txt_dataloader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args)











