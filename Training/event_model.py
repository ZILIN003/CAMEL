
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,TransformerEncoderLayer
from transformers import BertModel,AutoTokenizer
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.attention.cosine_attention import CosineAttention
from allennlp.nn.util import weighted_sum
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig, CLIPTokenizerFast
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.nn.util import masked_softmax
from allennlp.nn import util
import numpy as np
from collections import OrderedDict
from utils import mixup_data
import torchvision.ops.focal_loss as focal_loss

MODEL_TYPE = "openai/clip-vit-base-patch16"
LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']

class Unify_model_new(nn.Module):
    def __init__(self, bert_dir, clip_dir, config_para):
        print("Loading model")
        super().__init__()

        self.y_num1 = 9
        self.y_num2 = 9

        # init Bert
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        self.span_extractor = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x')
        self.T_sim_proj = nn.Linear(512, self.bert.config.hidden_size)
        self.V_sim_proj = nn.Linear(512, self.bert.config.hidden_size)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size + 768, self.y_num1)
        print('resume bert checkpoint from %s' % bert_dir)

        # init CLIP
        self.clip = CLIPModel.from_pretrained(MODEL_TYPE)
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_TYPE)
        self.clip_fc = nn.Linear(self.bert.config.hidden_size + 768, self.y_num2)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, 512 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(512 // 16, 512)),
            ("layernorm1", nn.LayerNorm(512)),
        ]))
        self.meta = False

        self.attention_block = AttentionBlock(n_head=12, d_model=768, d_k=64, d_v=64, d_in=768, d_hid=2048, dropout=0.1)

        # load CLIP
        if len(clip_dir)>0:
            checkpoint = torch.load(clip_dir, map_location='cpu')
            state_dict = checkpoint['model']
            self.clip.load_state_dict(state_dict)
            print('resume clip checkpoint from %s' % clip_dir)

        self.config_para = config_para

    def getName(self):
        return self.__class__.__name__

    def forward(self,type,input,device):
        if type=='ace':
            loss, _ = self.forward1(input,device)
        else:
            loss, _ = self.forward2(input, device, False, self.config_para.V_LS, self.config_para.V_Mixup)
        return loss

    # token-level sim
    def compute_logits(self, data_x, bert_mask, data_span, image):
        # torch.Size([10, 5, 512]) image

        outputs = self.bert(data_x, attention_mask=bert_mask)
        hidden_states = outputs[2]

        # torch.Size([10, 181, 768])
        temp = torch.cat(hidden_states[-1:], dim=-1)
        # remove cls token # torch.Size([10, 180, 768])reshape(10,180,768)
        temp = temp[:,1:,:].contiguous()
        temp = self.span_extractor(temp, data_span)

        # torch.Size([10, 5, 768])
        if self.meta:
            meta_output = self.meta_net(image)
            image = meta_output + image
        sim_img = self.T_sim_proj(image)

        # With attention block
        # ttt = self.T_attention_block(temp,sim_img,sim_img,False).squeeze(1)
        ttt = self.attention_block(temp, sim_img, sim_img, False).squeeze(1)

        # torch.Size([10, 180, 1280])
        temp = torch.cat([temp, ttt], dim=-1)
        logits = self.bert_fc(temp)
        return logits


    def forward1(self, input, device):

        data_x, bert_mask, data_span, sequence_mask, data_y, image, text, words, labels, images_num = input
        data_x, bert_mask, data_span, sequence_mask, data_y, image = data_x.to(device), bert_mask.to(
            device), data_span.to(device), sequence_mask.to(device), data_y.to(device), image.to(device)

        img_feature = self.clip.get_image_features(pixel_values=image)

        batch_size = data_x.shape[0]
        input_img_feat = img_feature.view(batch_size, 5, -1)

        # add cls token
        t1 = torch.tensor([101]*batch_size, device=device).long()
        t2 = torch.tensor([1] * batch_size, device=device).long()
        t1 = t1.reshape(batch_size,1)
        t2 = t2.reshape(batch_size, 1)
        data_x = torch.cat([t1,data_x], dim=1)
        bert_mask = torch.cat([t2, bert_mask], dim=1)

        logits = self.compute_logits(data_x, bert_mask, data_span, input_img_feat)
        ## Normal classification
        loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 is pad
        loss = loss_fct(logits.view(-1, self.y_num1), data_y.view(-1))

        return loss, logits

    def attention_T(self, matrix1, matrix2):
        a_norm = matrix1 / (
            matrix1.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix1.dtype)
        )
        b_norm = matrix2 / (
            matrix2.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix2.dtype)
        )
        similarities = torch.bmm(a_norm, b_norm.transpose(-1, -2))
        return torch.nn.functional.softmax(similarities, dim=-1)

    def attention_V(self, matrix1, matrix2):
        a_norm = matrix1 / (
            matrix1.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix1.dtype)
        )
        b_norm = matrix2 / (
            matrix2.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix2.dtype)
        )
        similarities = torch.mm(b_norm, a_norm.transpose(-1, -2))
        return torch.nn.functional.softmax(similarities, dim=-1)

    # Image attend to Text
    def forward2(self, input, device, M2E2, LS=0.0, Mixup=0.0):
        images, data_y, sentences = input[0].to(device),input[1].to(device),input[2]

        if Mixup>0:
            images, data_y, y_mix, lam = mixup_data(images, data_y, Mixup, device)

        img_feature = self.clip.get_image_features(pixel_values=images)
        batch_size = img_feature.shape[0]

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            for sen in sentences:
                length.append([flag,flag+len(sen)])
                sentence.extend(sen)
                flag +=len(sen)

            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)

            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature
            sim_img = self.V_sim_proj(img)

            # With attention block
            # cap = self.V_attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True).squeeze(1)
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat,True).squeeze(1)

            fc_input = torch.cat([cap, sim_img], dim=-1)
        else:

            # add cls with meta 768+512
            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)
            # fake_token = self.fake_token.unsqueeze(0).expand(batch_size, -1)
            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature

            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            # cap = self.V_attention_block(sim_img.unsqueeze(1), cap, cap,True).squeeze(1)
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True).squeeze(1)

            fc_input = torch.cat([cap, sim_img], dim=-1)

        logits = self.clip_fc(fc_input)

        if LS>0:
            loss_fct = CrossEntropyLoss(label_smoothing=LS)
        else:
            loss_fct = CrossEntropyLoss()

        if Mixup>0:
            loss = lam * loss_fct(logits.view(-1, self.y_num2), data_y.view(-1)) + (1 - lam) * loss_fct(logits.view(-1, self.y_num2), y_mix.view(-1))
        else:
            loss = loss_fct(logits.view(-1, self.y_num2), data_y.view(-1))

        return loss, logits


    @torch.no_grad()
    def predict_text_label(self, input, device):
        sequence_mask = input[3].to(device)

        loss, logits = self.forward1(input, device)

        classifications = torch.argmax(logits, -1)
        classifications = list(classifications.cpu().numpy())
        predicts = []
        for classification, mask in zip(classifications, sequence_mask):
            predicts.append(classification[:])

        return predicts, loss

    @torch.no_grad()
    def predict_img_label(self, input, device, M2E2):
        data_y = input[1].to(device)
        loss,logits = self.forward2(input, device, M2E2, LS=0.0, Mixup=0.0)

        classifications = torch.argmax(logits, -1)
        classifications = list(classifications.cpu().numpy())

        pred, ground = list(), list()

        acc_event = 0
        p = 0
        g = 0
        acc = 0
        for c, l in zip(classifications, data_y):
            if c == l:
                acc += 1
                if c != 0:
                    acc_event += 1
            if c != 0:
                p += 1
            if l != 0:
                g += 1

            pred.append(LABELS_list[c])
            ground.append(LABELS_list[l])

        return acc, p, g, loss, acc_event, pred, ground, logits

    @torch.no_grad()
    # indicate the fusion metheod before using this function
    def get_pnorm_logits(self, input, p, M2E2, device):
        images, data_y, sentences = input[0].to(device), input[1].to(device), input[2]
        img_feature = self.clip.get_image_features(pixel_values=images)
        batch_size = img_feature.shape[0]

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            for sen in sentences:
                length.append([flag, flag + len(sen)])
                sentence.extend(sen)
                flag += len(sen)

            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)

            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature
            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True).squeeze(1)

            # fc_input = torch.cat([cap, img], dim=-1)
            fc_input = torch.cat([cap, sim_img], dim=-1)
        else:

            # add cls with meta 768+512
            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)
            # fake_token = self.fake_token.unsqueeze(0).expand(batch_size, -1)
            if self.meta:
                meta_output = self.meta_net(img_feature)
                img = meta_output + img_feature
            else:
                img = img_feature

            sim_img = self.V_sim_proj(img)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            # cap = self.V_attention_block(sim_img.unsqueeze(1), cap, cap,True).squeeze(1)
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True).squeeze(1)

            # fc_input = torch.cat([cap, img], dim=-1)
            fc_input = torch.cat([cap, sim_img], dim=-1)

        def pnorm(weights, p):
            normB = torch.norm(weights, 2, 1)
            ws = weights.clone()
            for i in range(weights.size(0)):
                ws[i] = ws[i] / torch.pow(normB[i], p)
            return ws

        ws = pnorm(self.clip_fc.state_dict()['weight'], p)

        logits = torch.mm(fc_input, ws.t())
        return logits

    # Get fusion feature
    def get_fusion_feature(self, images, sentences, device, M2E2):

        img_feature = self.clip.get_image_features(pixel_values=images)
        img_feature_concatenate = self.clip.vision_model(pixel_values=images).last_hidden_state
        batch_size = img_feature.shape[0]

        if M2E2:
            length = list()
            flag = 0
            sentence = list()
            for sen in sentences:
                length.append([flag, flag + len(sen)])
                sentence.extend(sen)
                flag += len(sen)

            inputs = self.bert_tokenizer(sentence, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            txt_feature = last_hidden_state[:, 0, :].squeeze(1)
            for i, txt_index in enumerate(length):
                txt = torch.unsqueeze(txt_feature[txt_index[0]:txt_index[1]], dim=0)
                if i == 0:
                    input_txt_feat = txt
                else:
                    input_txt_feat = torch.cat((input_txt_feat, txt), 0)

            sim_img = self.V_sim_proj(img_feature)

            # With attention block
            cap = self.attention_block(sim_img.unsqueeze(1), input_txt_feat, input_txt_feat, True)
        else:

            inputs = self.bert_tokenizer(sentences, padding=True, return_tensors="pt").to(device)
            last_hidden_state = self.bert(**inputs).last_hidden_state
            cap = last_hidden_state[:, 0, :].squeeze(1)

            sim_img = self.V_sim_proj(img_feature)

            # With attention block
            cap = cap.unsqueeze(0)
            cap = cap.expand(batch_size, cap.shape[1], cap.shape[2])
            cap = self.attention_block(sim_img.unsqueeze(1), cap, cap, True)

        fusion_feature = torch.cat([cap, sim_img.unsqueeze(1), img_feature_concatenate[:, 1:,:]], dim=1)

        return fusion_feature



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

# imgtxt
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.cls = nn.Linear(d_model, n_head * d_k, bias=False)
        self.token = nn.Linear(d_model, n_head * d_k, bias=False)
        self.img = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, flag, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if flag:
            q = self.img(q).view(sz_b, len_q, n_head, d_k)
            k = self.cls(k).view(sz_b, len_k, n_head, d_k)
            v = self.cls(v).view(sz_b, len_v, n_head, d_v)
        else:
            q = self.token(q).view(sz_b, len_q, n_head, d_k)
            k = self.img(k).view(sz_b, len_k, n_head, d_k)
            v = self.img(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.PositionwiseFeedForward = PositionwiseFeedForward(d_in, d_hid, dropout)

    def forward(self, q, k, v, flag=True):
        output1, attn = self.MultiHeadAttention(q, k, v, flag)
        output = self.PositionwiseFeedForward(output1)

        return output
