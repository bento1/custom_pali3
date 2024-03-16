from CustomPALI3.SummaryChartDataset import SummaryChartDataset
from CustomPALI3.CustomPALI3 import CustomPALI3Config,CustomPALI3
from transformers import T5Tokenizer
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
from pytorch_model_summary import summary
import numpy as np
import gc

tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-base", bos_token = '<s>',add_bos_token = True)

config=CustomPALI3Config(version=1,model_name='test',
                    dim=1024,enc_num_tokens=32100,enc_max_seq_len=1024,
                    dec_num_tokens=32100,dec_max_seq_len=1024,enc_depth=4,enc_heads=8,dec_depth=4,dec_heads=8,seq_len=1024
                    ,device='mps',vit_fix=False)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model=CustomPALI3(config)
model=model.from_pretrained("/Users/dongunyun/study/datascience/chart2text/PALI3/output_temp")
model.model.seq_len=128

from torchvision import transforms
from PIL import Image
img=Image.open('test.png').convert("RGB").resize((336,336),Image.Resampling.BILINEAR )
label='Explain this picture. <s>'

input_image_tensor = transforms.transforms.ToTensor()(img).squeeze(0)
input_image_tensor = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))(input_image_tensor).unsqueeze(0).to(device)
outputs = tokenizer(label, 
                    max_length=1024, 
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt",
                    return_length=True,
                    )['input_ids'].to(device)
# model.to(device)
gen_=model.generate(input_image_tensor,outputs)

for gen in gen_:
    result_text = tokenizer.decode(gen, skip_special_tokens=True)
    print(result_text)