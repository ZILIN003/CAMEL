
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import json
import argparse
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from models.blip import BLIP_Decoder
from transformers import AutoProcessor, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = 'BLIP'  #'BLIP' 'BLIP2' 'vit-gpt2' 'GIT' 'pix2struct' 'OFA'

# 67->80
sr_to_ace_mapping = {}
for line in open("Datasets/M2E2/imSitu/ace_sr_mapping.txt", 'r'):
    fields = line.strip().split()
    sr_to_ace_mapping[fields[0]] = fields[2].replace('||', ':').replace('|', '-')

my_list = ['destroying', 'saluting', 'subduing', 'gathering', 'ejecting', 'marching', 'aiming', 'confronting',
           'bulldozing']
for m in my_list:
    if m in sr_to_ace_mapping:
        sr_to_ace_mapping.pop(m)
aaa = {
    "floating": "Movement:Transport",
    "leading": "Contact:Meet",
    "cheering": "Conflict:Demonstrate",
    "restraining": "Justice:Arrest-Jail",
    "bulldozing": "Conflict:Attack",
    "mourning": "Life:Die",
    "tugging": "Conflict:Attack",
    'signing': 'Contact:Meet',
    'colliding': "Conflict:Attack",
    'weighing': "Movement:Transport",
    'sleeping': "Life:Die",
    'falling': "Life:Die",
    'confronting': 'Contact:Meet',
    'gambling': 'Transaction:Transfer-Money',
    'pricking': 'Transaction:Transfer-Money'
}

for a in aaa:
    b = aaa[a]
    sr_to_ace_mapping[a] = b


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    normalize,
])


class BLIP_eval_decoder(BLIP_Decoder):

    @torch.no_grad()
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0, num_return_sequences=1, top_k=0, prompt='a picture of'):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            image_embeds = image_embeds.repeat_interleave(num_return_sequences, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

        self.prompt = prompt
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 top_k=top_k,
                                                 num_return_sequences=num_return_sequences,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 num_return_sequences=num_return_sequences,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        captions = []

        temp_captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i in range(len(prompt)):
            captions.append(temp_captions[i*num_return_sequences:(i+1)*num_return_sequences])
        return captions



class imSitu_dataset(Dataset):
    def __init__(self, type):
        ann_file = 'Datasets/M2E2/imSitu/'
        if type == 'train':
            self.data = json.load(open(os.path.join(ann_file,"train.json"),'r'))
            dev_data = json.load(open(os.path.join(ann_file, "dev.json"), 'r'))
            self.data.update(dev_data)
            self.transform = transform

        else:
            self.data  = json.load(open(os.path.join(ann_file, "test.json"), 'r'))
            self.transform = transform

        self.nouns = json.load(open(os.path.join(ann_file, "imsitu_space.json"), 'r'))['nouns']
        self.imgs_names = list(self.data.keys())
        self.img_dir = ann_file+'of500_images_resized/'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_name = self.imgs_names[index]
        annotations = self.data[img_name]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        if Model == 'BLIP' or Model == 'BLIP2' or Model == 'GIT':
            img = self.transform(img)
        verb = annotations['verb']

        return img_name,img,verb

class m2e2_img_Dataset(Dataset):
    def __init__(self):
        self.labels = []
        self.images = []
        event_dict = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_only_event.json').read())
        event_dict1 = json.loads(open('Datasets/M2E2/voa/m2e2_annotations/image_multimedia_event.json').read())
        event_dict.update(event_dict1)
        image_dir = 'Datasets/M2E2/voa/m2e2_rawdata/image/image'
        all_images = [image_dir + '/' + f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        self.img_names = []
        for img in all_images:
            img_name = img[img.rfind('/')+1:]
            img_name = img_name[:img_name.rfind('.')]
            if img_name in event_dict:
                txt_label = event_dict[img_name]['event_type']
                label = txt_label
            else:
                label = 'None'
            self.img_names.append(img_name)
            self.images.append(img)
            self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if Model == 'BLIP' or Model == 'BLIP2' or Model == 'GIT':
            image = self.transform(image)
        img_name = self.img_names[index]
        label = self.labels[index]
        return img_name, image, label


def my_collate(batch):

    img_name_list, img_list, verb_list = list(), list(), list()


    for img_name,img,verb in batch:
        img_name_list.append(img_name)
        img_list.append(img)
        verb_list.append(verb)

    real_batch = (img_name_list,img_list,verb_list)

    return real_batch

def main(device, dataset, caption_output_path):

    params_imSitu = {
        'batch_size': 128,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True
    }
    checkpoint_path = "Code/Event/BLIP/checkpoint/model_large_caption.pth"
    model = BLIP_eval_decoder(vit='large')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()

    model.to(device)
    dataloader = DataLoader(dataset, **params_imSitu)
    caption_dict = {}


    for name, images, verb in tqdm(dataloader):
        images= images.to(device)
        preds = model.generate(images, sample=True, top_p=0.9, max_length=50, min_length=5, num_return_sequences=10)

        for n, v, caption in zip(name, verb, preds):
            caption_dict[n] = {'verb': v, 'cap': caption}

    with open(caption_output_path, 'w') as f:
        json.dump(caption_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    train_dataset_imSitu = imSitu_dataset('train')
    path_train = 'Unify/caption/blip_m2e2_train.json'
    val_dataset_imSitu = imSitu_dataset('val')
    path_val = 'Unify/caption/blip_m2e2_val.json'
    m2e2_dataset_imSitu = m2e2_img_Dataset()
    path_test = 'Unify/caption/blip_m2e2_val.json'
    main(device, train_dataset_imSitu, path_train)
    main(device, val_dataset_imSitu, path_val)
    main(device, m2e2_dataset_imSitu, path_test)




