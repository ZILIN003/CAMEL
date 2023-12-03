
import numpy as np
import torch
from tqdm import tqdm
import pickle


def my_collate(batch):

    f = torch.FloatTensor
    l = torch.LongTensor

    data_x, data_span, data_y = list(), list(), list()
    images = list()
    text = list()
    words = list()
    labels = list()
    bert_mask, data_span_tensor, sequence_mask = list(), list(), list()
    images_num = list()
    flag=0

    for data in batch:
        data_x.append(data['data_x'].tolist())
        data_y.append(data['data_y'].tolist())
        text.append(data['text'])

        images.append(data['images'])

        words.append(data['words'])
        labels.append(data['labels'])

        bert_mask.append(data['bert_mask'].tolist())
        data_span_tensor.append(data['data_span_tensor'].tolist())
        sequence_mask.append(data['sequence_mask'].tolist())

        images_num.append((flag, flag+data['images_num']))
        flag = flag+data['images_num']

    images = torch.cat(images, 0)

    real_batch = (l(data_x),l(bert_mask).squeeze(1),l(data_span_tensor),l(sequence_mask).squeeze(1),l(data_y), f(images), text, words, labels,images_num)

    return real_batch

def m2e2_collate(batch):

    f = torch.FloatTensor
    l = torch.LongTensor

    data_y = list()
    images = list()
    text = list()

    for data in batch:
        images.append(data[0].unsqueeze(0))
        data_y.append(data[1])
        text.append(data[2])

    images = torch.cat(images, 0)

    real_batch = (f(images),l(data_y),text)

    return real_batch

def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


@torch.no_grad()
def get_m2e2_logits(model, data_loader, device, m2e2_img_list=None):
    model.eval()
    acc_total = 0
    p_total = 0
    g_total = 0
    acc_event_total = 0
    total = 0
    loss = 0

    pred_list = []
    ground_list = []
    logits_list = []

    for images, labels, sentences in tqdm(data_loader):
        total += len(labels)
        images = images
        labels = labels
        tmp_acc, tmp_p, tmp_g, tmp_loss, tmp_acc_event, pred, ground, logits = model.predict_img_label((images, labels, sentences),device,True)
        acc_total += tmp_acc
        p_total += tmp_p
        g_total += tmp_g
        acc_event_total += tmp_acc_event
        loss += tmp_loss.item() * len(labels)
        logits_list.append(logits)

        pred_list.extend(pred)
        ground_list.extend(ground)

    if p_total == 0:
        p = 0
    else:
        p = acc_event_total / p_total
    r = acc_event_total / g_total
    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    print(p)
    print(r)
    print(f1)

    logits_list = torch.cat(logits_list,dim=0)
    print(logits_list.shape)

    Label = {}
    for p, g, img, logit in zip(pred_list, ground_list, m2e2_img_list,logits_list):
        img_name = img.split('/')[-1][:-4]
        Label[img_name] = (p, g, logit)

    with open("/home/duzilin/Code/Event/BLIP/Unify/output/event/refine_VEE/result0/before_refine.pkl", "wb") as f:
        pickle.dump(Label, f)


##############
@torch.no_grad()
def accuracy_swig_bbox(pred_bbox, pred_bbox_conf, num_roles, bbox_exist, target):
    # device = pred_bbox.device
    w, h = target['width'], target['height']
    shift_0, shift_1, scale = target['shift_0'], target['shift_1'], target['scale']
    gt_bbox = target['ground_truth']

    # print(type(pred_bbox[0][2]))
    # print(type(shift_1))
    # print(type(gt_bbox))
    # print(type(scale))
    # print("####")

    # print(pred_bbox)
    # print(gt_bbox)
    # print("*****")

    if num_roles > 0:
        # convert predicted boxes
        for i in range(num_roles):
            pred_bbox[i][0] = max(pred_bbox[i][0] - shift_1, 0)
            pred_bbox[i][1] = max(pred_bbox[i][1] - shift_0, 0)
            pred_bbox[i][2] = max(pred_bbox[i][2] - shift_1, 0)
            pred_bbox[i][3] = max(pred_bbox[i][3] - shift_0, 0)
            # locate predicted boxes within image (processing w/ image width & height)
            pred_bbox[i][0] = min(pred_bbox[i][0], w)
            pred_bbox[i][1] = min(pred_bbox[i][1], h)
            pred_bbox[i][2] = min(pred_bbox[i][2], w)
            pred_bbox[i][3] = min(pred_bbox[i][3], h)
        pred_bbox /= scale

    # convert target boxes

    # gt_bbox[:, 0] = gt_bbox[:, 0] - shift_1
    # gt_bbox[:, 1] = gt_bbox[:, 1] - shift_0
    # gt_bbox[:, 2] = gt_bbox[:, 2] - shift_1
    # gt_bbox[:, 3] = gt_bbox[:, 3] - shift_0
    # gt_bbox /= scale

    predict = 0
    ground = 0
    acc = 0
    for i in range(num_roles):
        if gt_bbox[i][0]>=0:
            ground +=1
        if pred_bbox_conf[i] >= 0:
            predict += 1
            if bbox_exist[i] and bb_intersection_over_union(pred_bbox[i], gt_bbox[i]):
                acc+=1

    return predict, ground, acc

# def get_ground_truth(target, SWiG_json, idx_to_role, device):
#     bboxes = []
#     img_name = target['img_name']
#     for role in target['roles']:
#         role_name = idx_to_role[role]
#         if SWiG_json[img_name]["bb"][role_name][0] == -1:
#             bboxes.append(torch.tensor([-1, -1, -1, -1], device=device))
#         else:
#             b = [int(i) for i in SWiG_json[img_name]["bb"][role_name]]
#             bboxes.append(torch.tensor(b, device=device))
#     bboxes = torch.stack(bboxes)
#     return bboxes


def bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(xB - xA, 0) * max(yB - yA, 0)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = max(boxAArea + boxBArea - interArea, 1e-4)
        iou = interArea / float(unionArea)
        if iou >= 0.5:
            return True
        else:
            return False

############## box_ops
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def swig_box_xyxy_to_cxcywh(x, mw, mh, device=None, gt=False):
    # if x0, y0, x1, y1 == -1, -1, -1, -1, then cx and cy are -1.
    # so we can determine the existence of groundings with b[:, 0] != -1 (for gt)
    x0, y0, x1, y1 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    cx = ((x0 + x1) / 2).unsqueeze(1)
    cy = ((y0 + y1) / 2).unsqueeze(1)
    w = ((x1 - x0)).unsqueeze(1)
    h = ((y1 - y0)).unsqueeze(1)
    b = torch.cat([cx, cy, w, h], dim=1)
    if device is None:
        if gt:
            b[b[:,0] != -1] /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
            b[b[:,0] == -1] = -1
        else:
            b /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
    else:
        if gt:
            b[b[:,0] != -1] /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
            b[b[:,0] == -1] = -1
        else:
            b /= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
    return b

def swig_box_cxcywh_to_xyxy(x, mw, mh, device=None, gt=False):
    # if x_c, y_c, w, h == -1, -1, -1, -1, then x0 < 0.
    # so we can determine the existence of groundings with b[:, 0] < 0 (for gt)
    if device is None:
        if gt:
            x[x[:,0] != -1] *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
        else:
            x *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32)
    else:
        if gt:
            x[x[:,0] != -1] *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
        else:
            x *= torch.tensor([mw, mh, mw, mh], dtype=torch.float32, device=device)
    x_c, y_c, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    x0 = (x_c - 0.5 * w).unsqueeze(1)
    y0 = (y_c - 0.5 * h).unsqueeze(1)
    x1 = (x_c + 0.5 * w).unsqueeze(1)
    y1 = (y_c + 0.5 * h).unsqueeze(1)
    b = torch.cat([x0, y0, x1, y1], dim=1)
    if gt:
        b[b[:,0] < 0] = -1
    return b

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

from typing import Optional, List
from torch import Tensor

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))


        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def get_anchor(coordinate, return_num=3,img_size=224, patch_size=16):
    [x1,y1,x2,y2] = coordinate
    length = int(img_size/patch_size)

    def get_patch_num(c1,c2):
        num_x = int(c1 / patch_size) + 1
        num_y = int(c2 / patch_size) * length
        return num_x+num_y

    x_a = (x1 + x2) / 2
    y_a = (y1 + y2) / 2

    num1 = get_patch_num(x1, y1)
    num2 = get_patch_num(x2, y2)
    num3 = get_patch_num(x_a, y_a)

    if return_num==1:
        return torch.tensor([num3])
    elif return_num==2:
        return torch.tensor([num1, num2])
    elif return_num==3:
        return torch.tensor([num1, num2, num3])