from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import time
import urllib.request
from nltk.corpus import wordnet
import nltk
import requests
import numpy as np
from urllib.request import urlopen
import io
import os
from functools import wraps
import errno
import signal

import os
import json
def load_dataset():
    #pew_dataset 1000ea
    #statista_dataset 1000ea
    #scicap_data nosubfig 1000ea
    #simulated_scatter 2000ea
    #test 100
    pew_dataset_root_path='../dataset/pew_dataset_reduced'
    statista_dataset_root_path='../dataset/statista_dataset_reduced'
    scicap_data_root_path='../dataset/scicap_data_reduced'
    simulated_scatter_root_path='../dataset/simulated_scatter_dataset'
    simulated_trace_root_path='../dataset/simulated_trace_dataset'
    simulated_bar_root_path='../dataset/simulated_bar_dataset'
    simulated_comb_root_path='../dataset/simulated_comb_dataset'
    train_dataset=[]
    valid_dataset=[]
    test_dataset=[]
    #####################################################################################
    #####################################################################################

    for r_path in [pew_dataset_root_path,statista_dataset_root_path]:
        imagpath=os.path.join(r_path,'dataset','imgs')
        capspath=os.path.join(r_path,'dataset','captions')

        fileEx = r'.png'
        file_list = [file.split('.')[0] for file in os.listdir(imagpath) if file.endswith(fileEx)]
        train_file_list=file_list[:1000]
        valid_file_list=file_list[1000:1100]
        test_file_list=file_list[1100:1200]

        def readTxt(path):
            # readline_all.py
            f = open(path, 'r')
            result=""
            while True:
                line = f.readline()
                if not line: break
                result+=line+' '
            f.close()
            return result

        for filename in train_file_list:
            image_path=os.path.join(imagpath,f'{filename}.png')
            cap_path=os.path.join(capspath,f'{filename}.txt')
            train_dataset.append({'image':image_path,'text':readTxt(cap_path)})

        for filename in valid_file_list:
            image_path=os.path.join(imagpath,f'{filename}.png')
            cap_path=os.path.join(capspath,f'{filename}.txt')
            valid_dataset.append({'image':image_path,'text':readTxt(cap_path)})

        for filename in test_file_list:
            image_path=os.path.join(imagpath,f'{filename}.png')
            cap_path=os.path.join(capspath,f'{filename}.txt')
            test_dataset.append({'image':image_path,'text':readTxt(cap_path)})
    #####################################################################################
    #####################################################################################

    capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','train')
    fileEx = r'.json'
    file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    train_file_list=[]

    for filename in file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object:
            if json_object["contains-subfigure"]==False:
                train_file_list.append(filename)
                if len(train_file_list)==1000:
                    break

    imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','train')
    for filename in train_file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
            image_file_name=json_object['figure-ID']
            image_path=os.path.join(imgpath,image_file_name)
            train_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})

    capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','val')
    fileEx = r'.json'
    file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    valid_file_list=[]

    for filename in file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object:
            if json_object["contains-subfigure"]==False:
                valid_file_list.append(filename)
                if len(valid_file_list)==100:
                    break

    imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','val')
    for filename in valid_file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
            image_file_name=json_object['figure-ID']
            image_path=os.path.join(imgpath,image_file_name)
            valid_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})


    capspath=os.path.join(scicap_data_root_path,'SciCap-Caption-All','test')
    fileEx = r'.json'
    file_list = [file for file in os.listdir(capspath) if file.endswith(fileEx)]
    test_file_list=[]

    for filename in file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object:
            if json_object["contains-subfigure"]==False:
                test_file_list.append(filename)
                if len(test_file_list)==100:
                    break

    imgpath=os.path.join(scicap_data_root_path,'SciCap-No-Subfig-Img','test')
    for filename in test_file_list:
        cap_path=os.path.join(capspath,filename)
        with open(cap_path) as f:
            json_object = json.load(f)
        if "contains-subfigure" in json_object and "figure-ID" in json_object and "1-lowercase-and-token-and-remove-figure-index" in json_object:
            image_file_name=json_object['figure-ID']
            image_path=os.path.join(imgpath,image_file_name)
            test_dataset.append({'image':image_path,'text':json_object['1-lowercase-and-token-and-remove-figure-index']['caption']})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_scatter_root_path,'data','train')
    imagepath=os.path.join(simulated_scatter_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_scatter_root_path,'data','valid')
    imagepath=os.path.join(simulated_scatter_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_scatter_root_path,'data','test')
    imagepath=os.path.join(simulated_scatter_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_trace_root_path,'data','train')
    imagepath=os.path.join(simulated_trace_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_trace_root_path,'data','valid')
    imagepath=os.path.join(simulated_trace_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_trace_root_path,'data','test')
    imagepath=os.path.join(simulated_trace_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_bar_root_path,'data','train')
    imagepath=os.path.join(simulated_bar_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_bar_root_path,'data','valid')
    imagepath=os.path.join(simulated_bar_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_bar_root_path,'data','test')
    imagepath=os.path.join(simulated_bar_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    #####################################################################################
    #####################################################################################
    capspath=os.path.join(simulated_comb_root_path,'data','train')
    imagepath=os.path.join(simulated_comb_root_path,'image','train')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            train_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_comb_root_path,'data','valid')
    imagepath=os.path.join(simulated_comb_root_path,'image','valid')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            valid_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})

    capspath=os.path.join(simulated_comb_root_path,'data','test')
    imagepath=os.path.join(simulated_comb_root_path,'image','test')
    fileEx = r'.json'
    file_list = [file.split('.')[0] for file in os.listdir(capspath) if file.endswith(fileEx)]

    for filename in file_list:
        image_path=os.path.join(imagepath,f'{filename}.png')
        cap_path=os.path.join(capspath,f'{filename}.json')
        with open(cap_path) as f:
            json_object = json.load(f)
        if "description_rewrite" in json_object:
            test_dataset.append({'image':image_path,'text':json_object['description_rewrite'],'origin_text':cap_path})


    dataset=dict()
    dataset['train']=train_dataset
    dataset['valid']=valid_dataset
    dataset['test']=test_dataset
    return dataset
# https://huggingface.co/keras-io/ocr-for-captcha

nltk.download('wordnet')
nltk.download('punkt')
class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator

class SummaryChartDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_length: int,
        tokenizer,
        prompt_end_token,
        split: str = "train",
        ignore_id: int = -100,
    ):
        super().__init__()
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.index=[]
        self.prompt_end_token = prompt_end_token 
        self.API_TOKEN='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'
        self.tokenizer = tokenizer
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.dataset = dataset
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        img_path = os.path.join(sample['image'])
        while not os.path.isfile(img_path):#없으면 True 있으면 False 가 되서 빠져나옴
            idx+=1
            if (idx-1)>=len(self.dataset):
                idx=0
            sample = self.dataset[idx]
            img_path = os.path.join(sample['image'])
        img = Image.open(img_path).convert("RGB")
        
        input_image_tensor = transforms.transforms.ToTensor()(img).squeeze(0)
        # input_image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image_tensor)
        input_image_tensor = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))(input_image_tensor)
        input_image_tensor = transforms.Resize((336,336))(input_image_tensor)

        if 'origin_text' in sample:
            cap_path=sample['origin_text']
            with open(cap_path) as f:
                json_object = json.load(f)
            if 'description' in json_object:
                if "This scatterplot is a chart showing the distribution of values over time" in json_object["description"]:
                    query="describe this chart about trend and its abnormalities <s>"
                elif "This scatterplot shows the clustering results of groups between" in json_object["description"]:
                    query="describe this chart about clustering and its abnormalities <s>"
                elif "This scatterplot is a graph that represents the relationship between" in json_object["description"]:
                    query="describe this chart about correlation and its abnormalities <s>"
                else:
                    query="describe this chart <s>"
            else:
                query="describe this chart <s>"
        else:
            query="describe this chart <s>"
        if self.split=='train' or self.split=='validation':
            if "You are a helpful" in  sample['text'] or "please don't share false information" in sample['text'] or "following sentence is" in sample['text']:
                if 'origin_text' in sample:
                    if 'description' in json_object:
                        query=query+json_object['description']+"</s>"
                    else:
                        query=query+"I don't have any idea in this chart"+"</s>"
                else:
                    query=query+"I don't have any idea in this chart"+"</s>"
            else:
                query=query+sample['text']+"</s>"
                
            outputs = self.tokenizer(query, 
                                        max_length=self.max_length, 
                                        padding="max_length", 
                                        truncation=False,
                                        return_tensors="pt",
                                        return_length=True,
                                        )
            outputs_input_ids=outputs['input_ids'].squeeze(0)
            outputs_mask=outputs['attention_mask'].squeeze(0).bool()
            return input_image_tensor,outputs_input_ids,outputs_mask
        else:
            outputs = self.tokenizer(query, 
                                    max_length=self.max_length, 
                                    padding="max_length", 
                                    truncation=False,
                                    return_tensors="pt",
                                    return_length=True,
                                    )
            outputs_input_ids=outputs['input_ids'].squeeze(0)
            return input_image_tensor,outputs_input_ids,img
        
    