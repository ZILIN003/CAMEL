import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler


HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 100
NUM_IMAGES_PER_PROMPT = 1
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
BATCH_SIZE = 24
LABELS= ['Transaction:Transfer-Money', 'Movement:Transport', 'Conflict:Attack',
                                                  'Contact:Meet', 'Justice:Arrest-Jail', 'Life:Die',
                                                  'Conflict:Demonstrate', 'Contact:Phone-Write']
# "CompVis/stable-diffusion-v1-4" "stabilityai/stable-diffusion-2-base" "stabilityai/stable-diffusion-2-1-base" "runwayml/stable-diffusion-v1-5"



class M2E2Dataset(Dataset):

    def __init__(self):
        self.data=[]
        self.data += json.load(open('Datasets/M2E2/voa/m2e2_annotations/text_multimedia_event.json', 'r'))
        self.data += json.load(open('Datasets/M2E2/voa/m2e2_annotations/text_only_event.json', 'r'))

        self.dict = {}
        for data_item in self.data:
            sentence_id = data_item['sentence_id']
            self.dict[sentence_id] = data_item
        self.data = list(self.dict.values())


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence_id = self.data[index]['sentence_id']
        sentence = self.data[index]['sentence']
        words = self.data[index]['words']
        if len(words)>60:
            sentence = (' ').join(words[:50])
        return {
            'sentence_id': sentence_id,
            'sentence': sentence
        }

def generate_M2E2():
    # Generate images for sentences in M2E2
    dataset = M2E2Dataset()
    print("Total number of data: {}".format(len(dataset)))
    test_params = {
            'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': 0
        }
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    dataloader = DataLoader(dataset,**test_params)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_auth_token="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",
        # scheduler = scheduler,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    for batch in tqdm(dataloader):
        sentence_id = batch['sentence_id']
        sentence = batch['sentence']

        with torch.autocast("cuda"):
            images = pipe(prompt = sentence, height=HEIGHT, width=WIDTH,
                          num_inference_steps=NUM_INFERENCE_STEPS,
                          num_images_per_prompt=NUM_IMAGES_PER_PROMPT).images

            for i, image in enumerate(images):
                image.save(f"Code/StableDiffusion/dif2//M2E2/{sentence_id[i]}.png")


class CustomDataset(Dataset):
    # 13362
    def __init__(self):

        ace_path = 'Datasets/M2E2/ace/'
        self.data_ = []
        self.data_ += json.load(open(ace_path + 'JMEE_train_filter_no_timevalue.json', 'r'))
        self.data_ += json.load(open(ace_path + 'JMEE_dev_filter_no_timevalue.json', 'r'))
        self.data_ += json.load(open(ace_path + 'JMEE_test_filter_no_timevalue.json', 'r'))

        num=0
        self.data = []
        for i,dict in enumerate(self.data_):
            target_sub_sentence_tuple = []
            non_target_sub_sentence_tuple = []
            words = dict['words']
            if dict["golden-event-mentions"]!=[]:
                num+=1
                labels = dict["golden-event-mentions"]
                for label in labels:
                    start = 99
                    end = 0
                    if label['trigger']['start'] < start:
                        start = label['trigger']['start']
                    if label['trigger']['end'] > end:
                        end = label['trigger']['end']
                    for a in label['arguments']:
                        if a['start'] < start:
                            start = a['start']
                        if a['end'] > end:
                            end = a['end']
                    if end-start>50:
                        end=start+50
                    if label['event_type'] in LABELS:
                        target_sub_sentence_tuple.append((start, end))
                    else:
                        non_target_sub_sentence_tuple.append((start, end))

                num = 0
                def gen_sub_sentence(start_l, end_l, word, flag):
                    nonlocal num
                    if num==0 and flag==3:
                        num=2
                    elif num==0:
                        num=4

                    sub_sentence = (' ').join(word[start_l:end_l])
                    self.data.append({"sub_sentence": sub_sentence, "sentence_id": i, "index":num, 'label': dict["golden-event-mentions"], 'position':(start_l,end_l)})
                    num+=1


                if len(target_sub_sentence_tuple)==1:
                    start, end = target_sub_sentence_tuple[0]
                    for t in range(6):
                        gen_sub_sentence(start, end, words, 1)
                elif len(target_sub_sentence_tuple)>=2:
                    for t1 in range(2):
                        start, end = target_sub_sentence_tuple[t1]
                        for t2 in range(3):
                            gen_sub_sentence(start, end, words, 2)

                elif len(non_target_sub_sentence_tuple)>0:
                    start, end = non_target_sub_sentence_tuple[0]
                    for t in range(3):
                        gen_sub_sentence(start, end, words, 3)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sentence_id = self.data[index]['sentence_id']
        sub_sentence= self.data[index]['sub_sentence']
        num= self.data[index]['index']

        return {
            'sentence_id': sentence_id,
            'sub_sentence':sub_sentence,
            'index': num
        }


def generate_ACE():

    dataset = CustomDataset()
    print("Total number of data with events: {}".format(len(dataset)))
    test_params = {
            'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': 8
    }

    dataloader = DataLoader(dataset,**test_params)
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_auth_token="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",
        # scheduler = scheduler,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    for batch in tqdm(dataloader):
        sentence_id = batch['sentence_id']
        sub_sentence = batch['sub_sentence']
        index = batch['index']

        with torch.autocast("cuda"):

            images = pipe(prompt = sub_sentence, height=HEIGHT, width=WIDTH,
                          num_inference_steps=NUM_INFERENCE_STEPS,
                          num_images_per_prompt=NUM_IMAGES_PER_PROMPT).images

            for i, image in enumerate(images):
                image.save(f"Code/StableDiffusion/dif2-1/%05d-%d-sub.png" % (sentence_id[i], index[i]))


if __name__ == '__main__':
    generate_ACE()
    generate_M2E2()
