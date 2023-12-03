
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

MODEL_TYPE = "openai/clip-vit-base-patch16"
LABELS_list= ['None','Conflict:Attack', 'Conflict:Demonstrate', 'Contact:Meet', 'Contact:Phone-Write', 'Justice:Arrest-Jail', 'Life:Die', 'Movement:Transport', 'Transaction:Transfer-Money']

class Unify_model_ARG(nn.Module):
    def __init__(self, bert_dir, clip_dir, y_num):
        print("Loading model")
        super().__init__()

        self.y_num = y_num

        # init Bert
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        self.span_extractor_x = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x')
        self.span_extractor_y = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='y')
        self.span_extractor_xy = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x,y')
        self.span_extractor_en = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size, combination='x+y')
        self.T_sim_proj = nn.Linear(512, self.bert.config.hidden_size)
        self.bert_fc = nn.Linear(self.bert.config.hidden_size * 3, self.y_num)
        # self.bert_fc = nn.Linear(self.bert.config.hidden_size * 2, self.y_num)
        print('resume bert checkpoint from %s' % bert_dir)

        # init CLIP
        self.clip = CLIPModel.from_pretrained(MODEL_TYPE)
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_TYPE)
        self.clip_fc = nn.Linear(self.bert.config.hidden_size + 768, self.y_num)
        self.V_sim_proj = nn.Linear(512, self.bert.config.hidden_size)

        self.attention_block = AttentionBlock(n_head=12, d_model=768, d_k=64, d_v=64, d_in=768, d_hid=2048, dropout=0.1)


    def getName(self):
        return self.__class__.__name__

    def forward(self,type,input,device):
        if type=='ace':
            loss, _ = self.forward1(input,device)
        else:
            loss, _ = self.forward2(input,device,False)
        return loss

    # token-level sim
    def compute_logits(self, data_x, bert_mask, data_span, input_img_feat):

        outputs = self.bert(data_x, attention_mask=bert_mask)
        bert_enc = outputs[0]  ### use it for multi-task learning?
        hidden_states = outputs[2]
        temp = torch.cat(hidden_states[-1:], dim=-1)
        sim_img = self.T_sim_proj(input_img_feat)

        f = torch.LongTensor

        fusion = []

        for x, span, img in zip(temp, data_span, sim_img):

            span_entity = f(span)[:,1:].unsqueeze(0).to('cuda')
            span_trigger = f(span)[:,:2].unsqueeze(0).to('cuda')

            # span_feature_entity = self.span_extractor_en(x.unsqueeze(0), span_entity).div(2.0)
            span_feature_entity = self.span_extractor_x(x.unsqueeze(0), span_entity)
            con_img_feat = self.attention_block(span_feature_entity, img.unsqueeze(0), img.unsqueeze(0), False)
            span_feature_trigger = self.span_extractor_x(x.unsqueeze(0), span_trigger)
            fusion.append(torch.cat([span_feature_trigger, span_feature_entity, con_img_feat], -1))
            # fusion.append(torch.cat([span_feature_trigger, span_feature_entity], -1))

        fusion = torch.cat(fusion, 1)
        fusion = fusion.view(-1, 768 * 3)
        # fusion = fusion.view(-1, 768 * 2)

        logits = self.bert_fc(fusion)

        return logits


    def forward1(self, input, device):

        data_x, bert_mask, data_span, data_y, image = input
        data_x, bert_mask, data_span, data_y, image = data_x.to(device), bert_mask.to(device), data_span, data_y, image.to(device)

        # image feature
        img_feature = self.clip.get_image_features(pixel_values=image)
        batch_size = data_x.shape[0]
        input_img_feat = img_feature.view(batch_size, 5, -1)

        logits = self.compute_logits(data_x, bert_mask, data_span,input_img_feat)

        yyy = list()
        for y in data_y:
            yyy.extend(y)
        yyy = torch.LongTensor(yyy).to('cuda')

        ## Normal classification
        loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 is pad
        loss = loss_fct(logits, yyy)

        return loss, logits


    @torch.no_grad()
    def predict_txt(self, input, device):

        data_x, bert_mask, data_span, image = input
        data_x, bert_mask, data_span, image = data_x.to(device), bert_mask.to(device), data_span, image.to(device)

        # image feature
        img_feature = self.clip.get_image_features(pixel_values=image)
        batch_size = data_x.shape[0]
        input_img_feat = img_feature.view(batch_size, 5, -1)

        logits = self.compute_logits(data_x, bert_mask, data_span, input_img_feat)

        return logits


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

