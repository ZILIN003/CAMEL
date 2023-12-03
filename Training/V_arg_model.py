
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor
# from towhee.models.coformer.utils import nested_tensor_from_tensor_list
# from towhee.models.coformer.backbone import build_backbone
# from towhee.models.coformer.transformer import build_transformer
# from towhee.models.coformer.config import _C
from event_model import Unify_model_new
from utils import accuracy_swig_bbox, generalized_box_iou, swig_box_cxcywh_to_xyxy, bb_intersection_over_union, get_anchor
from torch.nn import CrossEntropyLoss
import json
from V_arg_dataset import vidx_ridx,ROLES_list

# with open("/home/duzilin/Code/Event/BLIP/Unify/output/CKPT4/att/decouple/event/VEE_0.json") as f:
#     VEE = json.load(f)

class CoFormer(nn.Module):
    """CoFormer model for Grounded Situation Recognition"""

    def __init__(self, transformer, vidx_ridx, Unify_model_path):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()

        self.backbone = Unify_model_new(bert_dir="bert-base-cased", clip_dir='', config_para=None)
        if len(Unify_model_path)>0:
            self.backbone.load_state_dict(torch.load(Unify_model_path))
            print('resume bert Unify_model from %s' % Unify_model_path)

        self.transformer = transformer
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 13
        self.num_verb_tokens = 8

        self.input_proj =  nn.Linear(768, 512)

        # hidden dimension for tokens and image features
        hidden_dim = transformer.d_model

        # token embeddings
        self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)
        self.verb_token_embed = nn.Embedding(self.num_verb_tokens, hidden_dim)

        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Dropout(0.2),
                                            nn.Linear(hidden_dim * 2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.2),
                                                 nn.Linear(hidden_dim * 2, 1))


    def forward(self, data, targets, device=None, M2E2=False):
        """
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: verbs information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        images = data[0].to(device)
        batch_size = images.shape[0]
        sentences = data[1]
        aggregated_feature = self.backbone.get_fusion_feature(images, sentences, device, M2E2)

        max_num_roles = 3
        batch_bbox, batch_bbox_conf = [], []

        # model prediction
        for i in range(batch_size):
            outs = self.transformer(self.input_proj(aggregated_feature[i:i + 1]), self.verb_token_embed.weight, self.role_token_embed.weight,targets=targets[i])

            # output features & predictions
            final_rhs, selected_roles = outs[0], outs[1]
            num_selected_roles = len(selected_roles)

            bbox_pred = self.bbox_predictor(final_rhs).sigmoid()
            bbox_pred = F.pad(bbox_pred, (0, 0, 0, max_num_roles - num_selected_roles), mode='constant', value=0)[
                -1].view(1, max_num_roles, 4)
            bbox_conf_pred = self.bbox_conf_predictor(final_rhs)
            bbox_conf_pred = \
                F.pad(bbox_conf_pred, (0, 0, 0, max_num_roles - num_selected_roles), mode='constant', value=0)[-1].view(
                    1, max_num_roles, 1)

            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)

        # outputs
        out = {}
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)

        return out



class Transformer(nn.Module):
    """
    Transformer class.
    """
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_gaze_s2_dec_layers=3,
                 dim_feedforward=2048,
                 dropout=0.15,
                 activation="relu"
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Gaze-Step2 Transformer
        gaze_s2_dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.gaze_s2_dec = TransformerDecoder(gaze_s2_dec_layer, num_gaze_s2_dec_layers)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                aggregated_feature,
                verb_token_embed,
                role_token_embed,
                targets=None
               ):
        bs = aggregated_feature.shape[0]
        # Gaze-Step2 Transformer
        ## Deocder
        ### At training time, we assume that the ground-truth verb is given.
        ### At inference time, we use the predicted verb.
        #### For frame-role queries, we select the verb token embedding corresponding to the predicted verb.
        #### For frame-role queries, we select the role token embeddings corresponding to the roles associated with the predicted verb.

        selected_verb_token = verb_token_embed[targets['verbs']].view(1, -1)
        selected_roles = targets['roles']
        selected_role_tokens = role_token_embed[selected_roles]
        frame_role_queries = selected_role_tokens + selected_verb_token
        frame_role_queries = frame_role_queries.unsqueeze(1).repeat(1, bs, 1)
        # role_tgt = torch.zeros_like(frame_role_queries)

        final_rhs = self.gaze_s2_dec(frame_role_queries, self.ln1(aggregated_feature))
        final_rhs = self.ln2(final_rhs)
        final_rhs = final_rhs.transpose(1,2)

        return final_rhs, selected_roles


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder class.
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    """
    TransformerDecoderLayer class.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    # def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    #     return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        new_memory = memory.permute(1,0,2)

        tgt2 = self.multihead_attn(query=tgt2, key=new_memory,
                                   value=new_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])




def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SWiGCriterion(nn.Module):
    """
    Loss for CoFormer with SWiG dataset, and CoFormer evaluation.
    """

    def __init__(self, SWiG_json_train=None, SWiG_json_eval=None, idx_to_role=None):
        """
        Create the criterion.
        """
        super().__init__()

    def forward(self, outputs, targets, device, M2E2):
        """ This performs the loss computation, and evaluation of CoFormer.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        out = {}
        batch_size = outputs['pred_bbox_conf'].shape[0]
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs

        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], [], []
        for i in range(batch_size):
            if not M2E2:
                pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
                mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes'].to(device)
                cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
                num_roles = len(t['roles'])
                bbox_exist = target_bboxes[:, 0] != -1
                num_bbox = bbox_exist.sum().item()

                # bbox conf loss
                loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc[:num_roles],
                                                                    bbox_exist[:num_roles].float(), reduction='mean')
                batch_bbox_conf_loss.append(loss_bbox_conf)

                # bbox reg loss and giou loss
                if num_bbox > 0:
                    loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='none')
                    loss_giou = 1 - torch.diag(
                        generalized_box_iou(swig_box_cxcywh_to_xyxy(pb[bbox_exist], mw, mh, device=device),
                                                    swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], mw, mh,
                                                                                    device=device, gt=True)))
                    batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                    batch_giou_loss.append(loss_giou.sum() / num_bbox)

                # convert coordinates
                pb_xyxy = swig_box_cxcywh_to_xyxy(cloned_pb, mw, mh, device=device)

                predict, ground, acc = accuracy_swig_bbox(pb_xyxy.clone(), pbc, num_roles, bbox_exist, t)

            else:
                pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
                mw, mh, ground_truth = t['max_width'], t['max_height'], t['ground_truth']
                w, h = t['width'], t['height']
                shift_0, shift_1, scale = t['shift_0'], t['shift_1'], t['scale']

                predict, ground, acc = 0, 0, 0
                num_roles = len(t['roles'])
                # convert coordinates
                pb_xyxy = swig_box_cxcywh_to_xyxy(pb.clone(), mw, mh, device=device)

                for role_index in range(num_roles):
                    pb_xyxy[role_index][0] = max(pb_xyxy[i][0] - shift_1, 0)
                    pb_xyxy[role_index][1] = max(pb_xyxy[i][1] - shift_0, 0)
                    pb_xyxy[role_index][2] = max(pb_xyxy[i][2] - shift_1, 0)
                    pb_xyxy[role_index][3] = max(pb_xyxy[i][3] - shift_0, 0)
                    # locate predicted boxes within image (processing w/ image width & height)
                    pb_xyxy[role_index][0] = min(pb_xyxy[i][0], w)
                    pb_xyxy[role_index][1] = min(pb_xyxy[i][1], h)
                    pb_xyxy[role_index][2] = min(pb_xyxy[i][2], w)
                    pb_xyxy[role_index][3] = min(pb_xyxy[i][3], h)
                pb_xyxy /= scale

                for role_index in range(num_roles):
                    ground += len(ground_truth[role_index])
                    if pbc[role_index]>0:
                        predict+=1
                    else:
                        continue

                    for ground_bbox in ground_truth[role_index]:
                        if bb_intersection_over_union(pb_xyxy[i], ground_bbox):
                            acc+=1

        if not M2E2:
            if len(batch_bbox_loss) > 0:
                bbox_loss = torch.stack(batch_bbox_loss).mean()
                giou_loss = torch.stack(batch_giou_loss).mean()
            else:
                bbox_loss = torch.tensor(0., device=device)
                giou_loss = torch.tensor(0., device=device)

            bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()

            out['loss_bbox'] = bbox_loss
            out['loss_giou'] = giou_loss
            out['loss_bbox_conf'] = bbox_conf_loss

        return out, (predict, ground, acc)



class Classifier_VA(nn.Module):
    """CoFormer model for Grounded Situation Recognition"""

    def __init__(self, Unify_model_path,use_cap):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()

        self.backbone = Unify_model_new(bert_dir="bert-base-cased", clip_dir='', config_para=None)

        if len(Unify_model_path)>0:
            self.backbone.load_state_dict(torch.load(Unify_model_path))
            print('resume bert Unify_model from %s' % Unify_model_path)

        self.num_role_tokens = 14
        self.num_verb_tokens = 8
        self.verb_token_embed = nn.Embedding(self.num_verb_tokens, 768)
        self.use_cap = use_cap

        self.proj = nn.Sequential(nn.Linear(768 * 2, 2048),
                                  nn.Dropout(0.1),
                                  nn.Linear(2048, 768))

        self.fc = nn.Linear(768, self.num_role_tokens)

        self.proj_extra = nn.Linear(768*3, 768)


    def forward(self, data, targets, device=None):
        images = data[0].to(device)
        batch_size = images.shape[0]
        sentences = data[1]
        aggregated_feature = self.backbone.get_fusion_feature(images, sentences, device, False)
        loss_func = CrossEntropyLoss()
        logits = []
        loss = []

        # model prediction
        for i in range(batch_size):
            target = targets[i]
            anchors, roles = [], []
            for ann in target['ground_truth']:
                anchors.append(ann['anchor'])
                roles.append(ann['role'])
            roles = torch.tensor(roles).to(device)

            selected_verb_token = self.verb_token_embed.weight[target['verbs']].view(1, -1)

            feature = aggregated_feature[i:i + 1]
            feature_map = []

            for anchor in anchors:
                anchor_feature = torch.mean(feature[:,anchor + 1,:], dim=1)

                if not self.use_cap:
                    feature_map.append(anchor_feature)
                else:
                    anchor_feature1 = self.proj_extra(torch.cat((feature[:,0,:], feature[:,1,:], anchor_feature), dim=-1))
                    feature_map.append(anchor_feature1)

            feature_out = torch.cat(feature_map,dim=0)
            feature_final = torch.cat((feature_out, selected_verb_token.repeat(feature_out.shape[0],1)), dim=-1)

            logit = self.fc(self.proj(feature_final))
            logits.append(logit)

            loss.append(loss_func(logit,roles.view(-1)))

        return torch.stack(loss).mean(), logits

    @torch.no_grad()
    def predict(self, data, targets, device=None, dict_result={}, VEE=None):
        images = data[0].to(device)
        batch_size = images.shape[0]
        sentences = data[1]
        aggregated_feature = self.backbone.get_fusion_feature(images, sentences, device, True)
        p_num, g_num, a_num = 0, 0, 0

        # model prediction
        for i in range(batch_size):
            p_num_tmp, a_num_tmp, g_num_tmp = 0, 0, 0
            target = targets[i]
            shift_0, shift_1, scale = target['shift_0'].item(), target['shift_1'].item(), target['scale'].item()
            anchors = target['ground_truth']['bboxes']
            img_name = target['img_name']
            for arg_num in target['ground_truth']['argument'].values():
                g_num_tmp += len(arg_num)

            selected_verb_token = self.verb_token_embed.weight[target['verbs']].view(1, -1)

            feature = aggregated_feature[i:i + 1]
            feature_map = []
            # print("$$$$")
            # print(target)
            for anchor in anchors:

                x1 = anchor[0] * scale + shift_1
                y1 = anchor[1] * scale + shift_0
                x2 = anchor[2] * scale + shift_1 -1e-3
                y2 = anchor[3] * scale + shift_0 -1e-3
                anchor_point = get_anchor([x1,y1,x2,y2], return_num=3,img_size=224, patch_size=16)
                # print(anchor)
                # print([x1,y1,x2,y2])
                # print(anchor_point)

                anchor_feature = torch.mean(feature[:,anchor_point + 1,:], dim=1)

                if not self.use_cap:
                    feature_map.append(anchor_feature)
                else:
                    anchor_feature1 = self.proj_extra(torch.cat((feature[:,0,:], feature[:,1,:], anchor_feature), dim=-1))
                    feature_map.append(anchor_feature1)

            if len(feature_map)==0:
                continue

            feature_out = torch.cat(feature_map,dim=0)
            feature_final = torch.cat((feature_out, selected_verb_token.repeat(feature_out.shape[0],1)), dim=-1)

            logit = self.fc(self.proj(feature_final))

            classifications = torch.argmax(logit, -1)
            classifications = list(classifications.cpu().numpy())

            index_list = list(torch.topk(logit, k=self.num_role_tokens, dim=-1, largest=True)[1].cpu().numpy())

            for pred_label, pred_box, index in zip(classifications, target['ground_truth']['bboxes'], index_list):

                # Filter
                for verb_index in index:
                    if verb_index==13 or verb_index in vidx_ridx[target['verbs'].item()]:
                        pred_label = verb_index
                        break

                if pred_label ==13:
                    continue
                p_num_tmp+=1
                pred_role = ROLES_list[pred_label]
                if VEE[img_name][0]==VEE[img_name][1]:
                    if pred_role in target['ground_truth']['argument']:
                        for box in target['ground_truth']['argument'][pred_role]:
                            if bb_intersection_over_union(pred_box, box):
                                a_num_tmp+=1

            dict_result[img_name] = [a_num_tmp,p_num_tmp]
            p_num += p_num_tmp
            g_num += g_num_tmp
            a_num += a_num_tmp

        return p_num, g_num, a_num


