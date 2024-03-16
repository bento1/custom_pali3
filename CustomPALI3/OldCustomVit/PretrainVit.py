import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from CustomVit import CustomVit,CustomVitConfig
from zeta.nn.modules import SigLipLoss
from torchinfo import summary
from torch.utils.data import random_split
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os
import torch.utils.data as data_utils
def pretrain(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    scheduler,
    num_epochs,
    model_path,
):
    # scaler = GradScaler()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        step_num=0
        for images, labels in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            # with autocast(dtype=torch.bfloat16):
            #     embeddings = model(images)
            #     ans=torch.zeros_like(embeddings).to(device)
            #     for b_indx,la in enumerate(labels):
            #         ans[b_indx,la]=1
            #     loss = loss_fn(embeddings, ans, 1,0)
            #     # loss = loss_fn(embeddings, ans)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()
            # print(f"Epoch: {epoch+1}, Step: {step_num+1}, Train Loss: {loss.item()}")
            # step_num+=1
            embeddings = model(images)
            ans=torch.zeros_like(embeddings).to(device)
            for b_indx,la in enumerate(labels):
                ans[b_indx,la]=1
            # loss = loss_fn(embeddings, ans, 1,0)
            loss = loss_fn(embeddings, ans)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Step: {step_num+1}, Train Loss: {loss.item()}")
            step_num+=1
        scheduler.step()
        val_loss = validate(model, val_loader, loss_fn, device)
        print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            loss = loss_fn(embeddings, labels,1,0)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def split(full_dataset, val_percent, test_percent, random_seed=None):
    amount = len(full_dataset)

    test_amount = (
        int(amount * test_percent)
        if test_percent is not None else 0)
    val_amount = (
        int(amount * val_percent)
        if val_percent is not None else 0)
    train_amount = amount - test_amount - val_amount

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        (train_amount, val_amount, test_amount),
        generator=(
            torch.Generator().manual_seed(random_seed)
            if random_seed
            else None))
    
    return train_dataset, val_dataset, test_dataset
if __name__=="__main__":
    # Dataset and DataLoader
    ds_path='/Users/dongunyun/study/datascience/chart2text/PALI3/CustomPALI3/imagenet1k'
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((512,512))]
    )
    # dataset = datasets.ImageFolder(ds_path,train=True, download=False, transform=transform,)
    dataset = datasets.ImageFolder(ds_path, transform=transform,)
    indices = torch.arange(10000)
    dataset = data_utils.Subset(dataset, indices)
    train_dataset, val_dataset, test_dataset = split(dataset, 0.1, 0.1, 0)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # plt.figure()
    # count=1
    # for images, labels in train_loader:
    #     # print(labels)
    #     plt.subplot(1,2,count)
    #     plt.imshow(to_pil_image(images.squeeze(0)), cmap='gray')
    #     count+=1
    #     if count==3:
    #         break
    # plt.show()
    # Model, Optimizer, Scheduler and Loss
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    config=CustomVitConfig( version=1,  model_name = 'pretrain_imgnet1k',  image_size = 512, 
                        patch_size = 16,  dim= 2048, depth = 4,  head = 8,   num_class= 1000,  device=device , dtype=torch.float32)
    model = CustomVit(config)
    # model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # loss_fn = SigLipLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    summary(model,(10,3,512,512),device=device)
    # Pretrain the model
    model_path = "./model.pth"
    num_epochs = 100
    pretrain(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        scheduler,
        num_epochs,
        model_path,
    )