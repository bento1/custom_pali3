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
import copy

dataset_repo=[{'dataset':'timm/imagenet-12k-wds','config':'default','type':'vision-text'},
                {'dataset':'wikimedia/wikipedia','config':'20231101.en','type':'text'},
                {'dataset':'conceptual_captions','config':'labeled','type':'vision-text'},
                {'dataset':'poloclub/diffusiondb','config':'2m_random_1m','type':'vision-text'},
                {'dataset':'scientific_papers','config':'arxiv','type':'text'}
                ]
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
        dataset_repo,
        max_length: int,
        tokenizer,
        prompt_end_token,
        split: str = "train",
        ignore_id: int = -100,
    ):
        super().__init__()
        self.dataset_repo=dataset_repo
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.index=[]
        self.prompt_end_token = prompt_end_token 
        self.API_TOKEN='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'
        self.tokenizer = tokenizer
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.init_dataset()
        self.retry=0
    def query(self,API_URL,headers):
        response = requests.get(API_URL, headers=headers)
        return response.json()
    def getLengthDataset(self,dataset,config=None):
        headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
        setName=['train','validation']
        lengthResult=dict()
        if config is None:
            API_URL =f"https://datasets-server.huggingface.co/is-valid?dataset={dataset}"
        else:
            API_URL =f"https://datasets-server.huggingface.co/is-valid?dataset={dataset}&config={config}"
        data = self.query(API_URL,headers)#{'preview': True, 'viewer': False, 'search': False, 'filter': False}
        if 'viewer' in data:
            isValid=data['viewer']
        else:
            isValid=False
        if not isValid:
            return

        if config is None:
            API_URL =f"https://datasets-server.huggingface.co/size?dataset={dataset}"
        else:
            API_URL =f"https://datasets-server.huggingface.co/size?dataset={dataset}&config={config}"
        
        data = self.query(API_URL,headers)

        for typeData in setName:
            if 'splits' not in data['size']:
                continue
            for i in data['size']['splits']:
                if config is None:
                    if i['split']==typeData and i['dataset']==dataset:
                        lengthResult[typeData]=i['num_rows']
                        lengthResult['isSplit']=False
                else:
                    if i['split']==typeData and i['dataset']==dataset and i['config']==config:
                        lengthResult[typeData]=i['num_rows']
                        lengthResult['isSplit']=False
        if 'validation' not in lengthResult:
            trainNum=int(lengthResult['train']*0.8)
            validNum=lengthResult['train']-trainNum
            lengthResult['train']=trainNum
            lengthResult['validation']=validNum
            lengthResult['isSplit']=True

        return lengthResult 
    def init_dataset(self):
        dataset_length=0
        for dr in self.dataset_repo:
            dr['length']=self.getLengthDataset(dr['dataset'],dr['config'])
            if dr['length'] is None :
                continue
            dataset_length+=dr['length'][self.split]
            self.index.append({'accumulated_length':dataset_length,'dataset':dr['dataset'],'config':dr['config'],'type':dr['type']})
    
    def getImageNet(self,dataset,config,idx):
        try:
            headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
            API_URL =f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={idx}&length=1"
            data = self.query(API_URL,headers)
            my_url=data['rows'][0]['row']['jpg']['src']
            # data = urlopen(my_url)
            file_name=f"/Users/dongunyun/study/datascience/chart2text/PALI3/temp/getImageNet_{self.split}_{str(idx)}.png"
            urllib.request.urlretrieve(my_url, file_name)
            time.sleep(3)
            img_=Image.open(file_name).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
            img=copy.deepcopy(img_)
            img_.close()
            # response = requests.get(my_url)
            # if io.BytesIO(response.content)["name"]=="Error":
            #     return self.getConceptualCaption(dataset,config,idx+1)
            # image_bytes = io.BytesIO(response.content)
            # img=Image.open(image_bytes).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
        
            synset = wordnet.synset_from_pos_and_offset('n', int(data['rows'][0]['row']['json']['filename'].split('_')[0][1:]))
            input_='Explain this picture. <s> '
            label = input_+ [x.name() for x in synset.lemmas()][0]+'</s>'
            if os.path.isfile(file_name):
                os.remove(file_name)
        except Exception as e:
            print(e)
            return self.getImageNet(dataset,config,idx+1)
        return img,label
    
    def getConceptualCaption(self,dataset,config,idx):
        try:
            headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
            API_URL =f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={idx}&length=1"
            data = self.query(API_URL,headers)
            file_name=f"/Users/dongunyun/study/datascience/chart2text/PALI3/temp/conceptual_captions_{self.split}_{str(idx)}.png"
            urllib.request.urlretrieve(data['rows'][0]['row']['image_url'],file_name)
            time.sleep(3)
            # my_url=data['rows'][0]['row']['image_url']
            img_=Image.open(file_name).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
            img=copy.deepcopy(img_)
            img_.close()
            # response = requests.get(my_url)
            # if stresponse.=="Error":
            #     return self.getConceptualCaption(dataset,config,idx+1)
            # image_bytes = io.BytesIO(response.content)
            # img=Image.open(image_bytes).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
            input_='Explain this picture. <s> '
            label=input_+data['rows'][0]['row']['caption']+'</s>'
            if os.path.isfile(file_name):
                os.remove(file_name)
        except Exception as e:
            print(e)
            return self.getConceptualCaption(dataset,config,idx+1)
        return img,label
    
    def getDiffusiondb(self,dataset,config,idx):
        try:
            headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
            API_URL =f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={idx}&length=1"
            data = self.query(API_URL,headers)
            file_name=f"/Users/dongunyun/study/datascience/chart2text/PALI3/temp/diffustion_{self.split}_{str(idx)}.png"
            urllib.request.urlretrieve(data['rows'][0]['row']['image_url'], file_name)
            time.sleep(3)
            img_=Image.open(file_name).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
            img=copy.deepcopy(img_)
            img_.close()
            # my_url=data['rows'][0]['row']['image_url']
            # response = requests.get(my_url)
            # if io.BytesIO(response.content)["name"]=="Error":
            #     return self.getConceptualCaption(dataset,config,idx+1)
            # image_bytes = io.BytesIO(response.content)
            # img=Image.open(image_bytes).convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
            input_='Explain this picture. <s> '
            label=input_+ data['rows'][0]['row']['caption']+'</s>'
            if os.path.isfile(file_name):
                os.remove(file_name)
        except Exception as e:
            print(e)
            return self.getConceptualCaption(dataset,config,idx+1)
        return img,label

    def getWiki(self,dataset,config,idx):
        try:
            headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
            API_URL =f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={idx}&length=1"
            data = self.query(API_URL,headers)
            img=Image.fromarray(np.zeros((336,336,3),dtype=np.uint8),'RGB')
            label='<s> '+data['rows'][0]['row']['text']+'</s>'
        # stentences=nltk.tokenize.sent_tokenize(label)
        except Exception as e:
            print(e)
            return self.getWiki(dataset,config,idx+1)
        return img,label

    def getARXIV(self,dataset,config,idx):
        try:
            headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
            API_URL =f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=train&offset={idx}&length=1"
            data = self.query(API_URL,headers)
            img=Image.fromarray(np.zeros((336,336,3),dtype=np.uint8),'RGB')
            label='<s> '+data['rows'][0]['row']['article']+'</s>'
        # stentences=nltk.tokenize.sent_tokenize(label)
        except Exception as e:
            print(e)
            return self.getARXIV(dataset,config,idx+1)
        return img,label

    @timeout(10)
    def getData(self,idx):

        accum=0
        isFind=False
        accumulated_length=[0]+[s['accumulated_length'] for s in self.index]
        # 누적합사이 값을 이용하여 
        # 들어온 인덱스가 누적값 어디에 위치하는지 확인
        # 인덱스가 
        for al in range(len(accumulated_length)):
            if accumulated_length[al]<idx and accumulated_length[al+1]>=idx:
                poss=al
                accum=accumulated_length[al]
        sel_dataset_source=self.index[poss]
        length_info=[i for i in self.dataset_repo if i['dataset']==sel_dataset_source['dataset'] and i['config']==sel_dataset_source['config']][0]
        # 10
        # 20 
        # 7일경우, 7번째 골라야함. 그대로 
        # 15일경우 , 5번째를 골라야함. idx가 4여야함.
        if idx>accum:
            idx-=(accum+1) # 누적값만큼 뺀게 인덱스가 된다.
        # train 밖에 없는 소스의 경우 split 했기 때문에 
        # train 개수 123개  valid 23개 인경우
        # idx가 22로 계산되지만, 사실 145 이기때문에 + train 개수
        if self.split=='validation' and length_info['isSplit']:
            idx+=length_info['train']

        img,label=None,None
        dataset,config =sel_dataset_source['dataset'],sel_dataset_source['config']

        if sel_dataset_source['type']=='vision-text':
            if 'imagenet' in sel_dataset_source['dataset']:
                img,label=self.getImageNet(dataset,config,idx)
            elif 'conceptual_captions' in sel_dataset_source['dataset']:
                img,label=self.getConceptualCaption(dataset,config,idx)
            elif 'diffusiondb' in sel_dataset_source['dataset']:
                img,label=self.getDiffusiondb(dataset,config,idx)
            else:
                return torch.zeros((3,336,336)), torch.zeros(1024)       
        elif sel_dataset_source['type']=='text':
            if 'wikipedia' in sel_dataset_source['dataset']:
                img,label=self.getWiki(dataset,config,idx)
            elif 'scientific_papers' in sel_dataset_source['dataset']:
                img,label=self.getARXIV(dataset,config,idx)
            else:
                return torch.zeros((3,336,336)), torch.zeros(1024)       
        else:
            return torch.zeros((3,336,336)), torch.zeros(1024)       

        return img,label
    
    def __len__(self) -> int:
        
        return self.index[-1]['accumulated_length']

    def __getitem__(self, idx: int):
        img,label = self.getData(idx)
        
        input_image_tensor = transforms.transforms.ToTensor()(img).squeeze(0)
        # input_image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(input_image_tensor)
        input_image_tensor = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))(input_image_tensor)
        input_image_tensor = transforms.Resize((336,336))(input_image_tensor)
        outputs = self.tokenizer(label, 
                                    max_length=self.max_length, 
                                    padding="max_length", 
                                    truncation=False,
                                    return_tensors="pt",
                                    return_length=True,
                                    )
        image_batch = []
        input_batch = []
        attn_mask_batch = []
        
        batch_num=outputs['length']//self.max_length if outputs['length']%self.max_length==0 else outputs['length']//self.max_length+1
        outputs_input_ids=outputs['input_ids'].squeeze(0)
        outputs_mask=outputs['attention_mask'].squeeze(0)
        for i in range(batch_num):
            if (i+1)*self.max_length<outputs['length']:
                input_ids = outputs_input_ids[i*self.max_length:(i+1)*self.max_length]
                attn_mask = outputs_mask[i*self.max_length:(i+1)*self.max_length]
                attn_mask=attn_mask.bool()
            else:
                target= outputs_input_ids[i*self.max_length:]
                input_ids =torch.cat([target,torch.zeros(self.max_length-target.shape[0], dtype=torch.int64)], dim = 0)
                target_mask= outputs_mask[i*self.max_length:]
                attn_mask =torch.cat([target_mask,torch.zeros(self.max_length-target.shape[0], dtype=torch.int64)], dim = 0)
                attn_mask=attn_mask.bool()
            image_batch.append(input_image_tensor)
            input_batch.append(input_ids)
            attn_mask_batch.append(attn_mask)
        return {'image':image_batch ,"input_ids": input_batch,"attn_mask":attn_mask_batch}
        
    