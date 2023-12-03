
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json

import sys
sys.path.append('..')

import pickle

CKPT = 'Code/Event/BLIP/Unify/output/CKPT7/model_T_0.pth'
MODEL_TYPE = "openai/clip-vit-base-patch16"

LabelDict={'Transfer_Money':'Transaction:Transfer-Money', 'Transport': 'Movement:Transport', 'Attack':'Conflict:Attack',
                                                  'Meet':'Contact:Meet', 'Arrest_Jail':'Justice:Arrest-Jail', 'Die':'Life:Die',
                                                  'Demonstrate':'Conflict:Demonstrate', 'Phone_Write':'Contact:Phone-Write'}
LABELS= ['Transaction:Transfer-Money', 'Movement:Transport', 'Conflict:Attack',
                                                  'Contact:Meet', 'Justice:Arrest-Jail', 'Life:Die',
                                                  'Conflict:Demonstrate', 'Contact:Phone-Write']

def get_score(args, AE):


    with open("/home/duzilin/Code/Event/UniCL/code/Output/Score/score_CLIP.pkl", "rb") as f:
        score = pickle.load(f)

    mm_event = []
    mm_pair = []
    for line in open("Datasets/M2E2/voa/m2e2_annotations/crossmedia_coref.txt", 'r'):
        fields = line.strip().split()
        mm_event.append((fields[0],fields[1].split('.jpg')[0],fields[2]))
        mm_pair.append((fields[0],fields[1].split('.jpg')[0]))
    g_total = len(mm_event)

    with open(os.path.join("Code/Event/BLIP/Unify/output/unify0/decouple0/event/VEE_0.json"), 'r') as f:
        vee = json.load(f)
    with open(os.path.join("Code/Event/BLIP/Unify/output/unify0/event/TEE_4.json"), 'r') as f:
        tee = json.load(f)
    if AE:
        VAE = json.loads(open('Code/Event/BLIP/Unify/output/unify0/decouple0/event/VAE_0.json').read())
        TAE = json.loads(open('Code/Event/BLIP/Unify/output/unify0/event/TAE_4.json').read())


    text_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/text_multimedia_event.json').read())
    image_event = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_multimedia_event.json').read())

    temp_list = []
    for x in text_event:
        temp_list.append(x['sentence_id'])
    text_event_list = temp_list
    image_event_list = list(image_event.keys())


    flag = 0
    max_f1 = 0
    while flag<25:

        arg_acc1 = 0
        arg_pred1 = 0
        arg_acc2 = 0
        arg_pred2 = 0

        p_total = 0
        acc = 0
        score_threshold = flag
        for voa,file in score.items():
            sen_list = file['sen_id_list']
            img_list = file['img_id_list']
            matrix = file['score'].cpu()
            for i, sen in enumerate(sen_list):
                if sen not in text_event_list:
                    continue
                for j,img in enumerate(img_list):
                    img_name = img[:img.rfind('.jpg')]
                    # if img_name in vee_wrong:
                    #     continue
                    if img_name not in image_event_list:
                        continue
                    if vee[img_name][0]=='None':
                        continue
                    if matrix[j][i]<score_threshold:
                        continue
                    for event in list(set(tee[sen])):
                        p_total += 1
                        if event.split('B-')[1].replace('_', '-') in vee[img_name][0]:
                            if AE:
                                if img_name in VAE:
                                    arg_pred1 += VAE[img_name][1]
                                if sen in TAE:
                                    arg_pred2 += TAE[sen][1]

                            if (sen,img_name,vee[img_name][0]) in mm_event:
                                # print((sen,img_name,vee[img_name][0]))
                                acc+=1

                                if AE:
                                    if img_name in VAE:
                                        arg_acc1 += VAE[img_name][0]
                                    if sen in TAE:
                                        arg_acc2 += TAE[sen][0]
        print(flag)
        if AE:
            print("AE")
            print(arg_acc1)
            print(arg_pred1)
            print(arg_acc2)
            print(arg_pred2)


            arg_p = (arg_acc1+arg_acc2) / (arg_pred1+arg_pred2)
            arg_r = (arg_acc1+arg_acc2) / (658+307)
            arg_f1 = 2 * arg_p * arg_r / (arg_p + arg_r)

            print(arg_p)
            print(arg_r)
            print(arg_f1)

        print("EE")
        print(acc)
        print(p_total)
        print(g_total)
        p = acc / p_total
        r = acc / g_total
        f1 = 2 * p * r / (p + r)
        if max_f1<f1:
            max_f1=f1
        print("Precision: {}".format(p))
        print("Recall: {}".format(r))
        print("F1: {}".format(f1))
        print()
        flag += 1
    print(max_f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    get_score(args,True)
