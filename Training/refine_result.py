# Refer to https://github.com/jianliu-ml/Multimedia-EE/blob/main/code/textualEE/refine_result.py
import spacy
from spacy.tokens import Doc
import json
from conlleval import evaluate_conll_file
import random
import numpy as np


def read_m2e2():
    file1 = 'text_only_event.json'
    file2 = 'text_multimedia_event.json'
    temp = []
    sen_id_list = []

    dict = {}
    for f in [file1, file2]:
        xxx = json.loads(open(f).read())
        for x in xxx:
            pos = x['pos-tags']
            sen_id = x['sentence_id']
            if sen_id not in dict:
                dict[sen_id] = 1
                temp.append(pos)
                sen_id_list.append(sen_id)
    return temp, sen_id_list


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


def refine(input_result, refined_result):
    result = load_conll_data(input_result)
    poses, sen_id_list = read_m2e2()
    filename = refined_result
    fileout = open(filename, 'w')

    word_dic = {}
    word_corret = {}
    word_corret_bi = {}

    sen2type_dict = {}

    for res, pos in zip(result, poses):
        sens, gs, ps = res
        for idx, (w, g, p, s) in enumerate(zip(sens, gs, ps, pos)):
            word_dic.setdefault(w, 0)
            word_dic[w] += 1

            word_corret.setdefault(w, {})
            word_corret[w].setdefault('t', 0)
            word_corret[w]['t'] += 1
            if p == g:
                word_corret[w].setdefault('c', 0)
                word_corret[w]['c'] += 1

            w = (sens[idx - 1], sens[idx])
            word_corret_bi.setdefault(w, {})
            word_corret_bi[w].setdefault('t', 0)
            word_corret_bi[w]['t'] += 1
            if p == g:
                word_corret_bi[w].setdefault('c', 0)
                word_corret_bi[w]['c'] += 1

    zzz = [(w, word_corret[w].get('c', 0) / word_corret[w]['t']) for w in word_corret]
    zzz = sorted(zzz, key=lambda x: x[1], reverse=False)[:100]
    wrong_set = set([x[0] for x in zzz])

    zzz = [(w, word_corret_bi[w].get('c', 0) / word_corret_bi[w]['t']) for w in word_corret_bi]
    zzz = sorted(zzz, key=lambda x: x[1], reverse=False)[:1000]
    wrong_set_bi = set([x[0] for x in zzz])

    for res, pos, sen_id in zip(result, poses, sen_id_list):

        sen2type_dict.setdefault(sen_id, [])
        sens, gs, ps = res

        count = 0
        for i in range(len(sens)):
            if sens[i] == '"':
                count += 1
            if count > 0 and count % 2 == 1:
                ps[i] = 'O'

        for idx, (w, g, p, s) in enumerate(zip(sens, gs, ps, pos)):

            if w in wrong_set:
                p = 'O'

            if (sens[idx - 1], w) in wrong_set_bi:
                p = 'O'

            for i in range(max(0, idx - 10), min(len(sens), idx + 10)):
                if sens[i] == '“' or sens[i] == '”':
                    p = 'O'

            for i in range(max(0, idx - 9), min(len(sens), idx + 3)):
                if sens[i] == ':':
                    p = 'O'

            for i in range(max(0, idx - 3), min(len(sens), idx + 3)):
                if 'say' in sens[i] or 'avoid' in sens[i]:
                    p = 'O'

            if w == 'War' or w == 'deadly':
                p = 'O'

            if w == 'letter':
                p = 'O'

            if '-' in w:
                p = 'O'

            if w == 'talks' and not sens[idx - 1][0].isupper():
                p = 'O'

            if 'war-' in w:
                p = 'O'

            if 'email' in w:
                p = 'O'

            # if w == 'conflict' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'war' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'violence' and p == 'B-Attack' and s == 'NN':
            #     p = 'O'

            # if w == 'summit' and p == 'B-Meet' and s == 'NN':
            #     p = 'O'

            # if w == 'Summit' and p == 'B-Meet' and s == 'NNP':
            #     p = 'O'

            # if w == 'clashes' and p == 'B-Attack' and s == 'NNS':
            #     p = 'O'

            # if w == 'massacres' and s == 'NNS':
            #     p = 'O'

            # if w == 'suicide' and s == 'NN':
            #     p = 'O'

            # if w == 'shootings' and s == 'NNS':
            #     p = 'O'

            # if w == 'killings' and s == 'NNS':
            #     p = 'O'

            # if w == 'shelling' and s == 'VBG':
            #     p = 'O'

            # if w == 'murder' and s == 'NN':
            #     p = 'O'

            # if w == 'wars' and s == 'NNS':
            #     p = 'O'

            # if w == 'hostilities' and p == 'B-Attack' and s == 'NNS':
            #     p = 'O'

            # if w == 'die' and p == 'B-Die' and s == 'VB':
            #     p = 'O'

            # if w.lower().startswith('attack') and s.startswith('NN'):
            #     p = 'O'

            # if w.lower() == 'talks' and s.startswith('NN'):
            #     p = 'O'

            # if w.lower() == 'rally'  and s.startswith('NN'):
            #     p = 'O'

            if p != 'O':
                sen2type_dict[sen_id].append(p)

            print(w, g, p, file=fileout)
        print(file=fileout)
    fileout.close()

    with open(filename) as fout:
        res = evaluate_conll_file(fout)

    return res, sen2type_dict

