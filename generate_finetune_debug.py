from CustomPALI3.SummaryChartDatasetFineTune import load_dataset,SummaryChartDataset
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
model.model.seq_len=512
if __name__=='__main__':
    dataset=load_dataset()
    test_loader=SummaryChartDataset(dataset['test'],1024,tokenizer,'</s>','test')
    input_image_tensor,outputs_input_ids,img=test_loader.__getitem__(1600)
    # model.to(device)
    img.show()
    gen_=model.generate(input_image_tensor.unsqueeze(0).to(device),outputs_input_ids.unsqueeze(0).to(device))

    for gen in gen_:
        result_text = tokenizer.decode(gen, skip_special_tokens=True)
        print(result_text)