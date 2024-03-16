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

def finetune(
    model,
    args,
    train_loader,
    val_loader,
    optimizer,
    device,
    scheduler,
    # model_path,
):
    scaler = GradScaler()
    model_path=args['output_dir']
    best_val_loss = float("inf")
    for epoch in range(int(args['num_epochs'])):
        model.model.train()
        step_num=0
        for _ in range(int(args['max_steps'])):
            try:
                input_data = next(iter(train_loader))
                image,input_id,attn_mask=input_data
                image=image.to(device)
                input_id=input_id.to(device)
                attn_mask=attn_mask.to(device)
                optimizer.zero_grad()
                prev_dec_some_weight=model.model.pali_model.decoder.net.attn_layers.layers[0][1].to_out.weight[0,0].item()
                prev_enc_some_weight=model.model.pali_model.encoder.attn_layers.layers[1][1].ff[0][0].weight[0,0].item()
                prev_vit_some_weight=model.model.vit_model.model.model.vision_model.encoder.layers[-1].mlp.fc1.weight[0,0].item()

                logits, loss = model(img=image,prompt=input_id,output=input_id,mask=attn_mask)
                eta = 10 # maximum loss value
                loss = torch.clamp(loss,max = eta)
                if (loss.isnan() != 0): # no nan values
                    print('loss is nan value')
                    loss=prev_loss
                prev_loss=loss
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                scheduler.step()
                prev_dec_some_weight2=model.model.pali_model.decoder.net.attn_layers.layers[0][1].to_out.weight[0,0].item()
                prev_enc_some_weight2=model.model.pali_model.encoder.attn_layers.layers[1][1].ff[0][0].weight[0,0].item()
                prev_vit_some_weight2=model.model.vit_model.model.model.vision_model.encoder.layers[-1].mlp.fc1.weight[0,0].item()
                print('diff_vit_enc:',prev_vit_some_weight2-prev_vit_some_weight,'diff_text_enc:',prev_enc_some_weight2-prev_enc_some_weight,'diff_text_dec',prev_dec_some_weight2-prev_dec_some_weight)
                print(f"Epoch: {epoch+1}, Step: {step_num+1}, Train Loss: {loss}")
                step_num+=1

                if step_num%100==0 and step_num!=0:
                    save_checkpoint(model,model_path+'_temp_finetune')

            except Exception as e:
                print('occurs error : ',e)
                continue

        val_loss = validate(model, val_loader, device,args)

        print(f"Epoch: {epoch+1}, Train Loss: {loss}, Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model,model_path)

def validate(model, dataloader, device,args):
    model.model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(int(args['valid_steps'])):
            try:
                input_data = next(iter(dataloader))
                image,input_id,attn_mask=input_data
                image=image.to(device)
                input_id=input_id.to(device)
                attn_mask=attn_mask.to(device)
                logits, loss = model(img=image,prompt=input_id,output=input_id,mask=attn_mask)
                total_loss += loss
            except:
                continue
    return total_loss / len(dataloader)

def save_checkpoint(model,save_path):
    model.save_pretrained(save_path, from_pt=True)

def getParameters(model):
    count=0
    def mul(list_):
            init=1
            for i in list_:
                    init*=i
            return init
    for name, param in model.vit_model.named_parameters():
            count+=mul(np.array(param.size()).tolist())
    print('vit_model',count/1000000,"M")
    count=0
    for name, param in model.pali_model.named_parameters():
            count+=mul(np.array(param.size()).tolist())
    print('pali_model',count/1000000,"M")

if __name__=='__main__':
    dataset=load_dataset()
    
    args={
        'output_dir':'/Users/dongunyun/study/datascience/chart2text/PALI3/output',
        'lr':5e-5,
        'max_steps':len(dataset['train']),
        'valid_steps':1e2,
        'num_epochs':100,
        'batch_size':3,
        'max_epochs':100,
        "warmup_steps":100,
        'num_workers':1,
        'num_nodes':1,
        }
    
    tokenizer=T5Tokenizer.from_pretrained("google/flan-t5-base", bos_token = '<s>',add_bos_token = True)
    train_loader=DataLoader(SummaryChartDataset(dataset['train'],1024,tokenizer,'</s>','train'), batch_size=args['batch_size'], shuffle=True, num_workers=1)
    val_loader=DataLoader(SummaryChartDataset(dataset['valid'],1024,tokenizer,'</s>','validation'), batch_size=args['batch_size'], shuffle=True, num_workers=1)
    config=CustomPALI3Config(version=1,model_name='test',
                        dim=1024,enc_num_tokens=32100,enc_max_seq_len=1024,
                        dec_num_tokens=32100,dec_max_seq_len=1024,enc_depth=4,enc_heads=8,dec_depth=4,dec_heads=8,seq_len=1024
                        ,device='mps',vit_fix=False)
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model=CustomPALI3(config)
    model=model.from_pretrained("/Users/dongunyun/study/datascience/chart2text/PALI3/output_temp")
    model.config.ctc_zero_infinity = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.1)
    summary(model,torch.zeros((1,3,336,336)).to(device=device,dtype=torch.long),
                            torch.zeros((1,1024)).to(device=device,dtype=torch.int32),
                            torch.zeros((1,1024)).to(device=device,dtype=torch.int32),
                            torch.ones(1, 1024).bool().to(device=device),
                            show_input=True, print_summary=True,)
    getParameters(model.model)
    finetune(
        model,
        args,
        train_loader,
        val_loader,
        optimizer,
        device,
        scheduler,
    )