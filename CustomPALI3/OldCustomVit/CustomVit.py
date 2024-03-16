from pali3.main import VitModel
from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn.functional as F


class CustomVitConfig(PretrainedConfig):
    model_type = "mymodel"
    def __init__(
        self,
        version = 1,
        model_name='',
        image_size=512,
        patch_size=256,
        dim=512,depth=12,
        heads=32, num_class=1000,
        device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
        dtype=torch.bfloat16,
        **kwargs,
    ):
        self.version = version
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.device = device
        self.dtype = dtype
        self.num_class=num_class
        super().__init__(**kwargs)

class CustomVit(PreTrainedModel):
    config_class = CustomVitConfig

    def __init__(self, config:CustomVitConfig):
        super().__init__(config)
        self.model = CustomVit_(
            model_name=config.model_name,
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            num_class=config.num_class,
            device=config.device,
            dtype=config.dtype
            )
        # self.model=self.model.to(config.device)
    def forward(self, img):
        return self.model(img)

class CustomVit_(VitModel,torch.nn.Module):
    def __init__(self,model_name,image_size,patch_size,
        dim,depth,heads,num_class,
        device,dtype
        ):
        # super().__init__() ## 이거 추가
        # super(torch.nn.Module, self).__init__() ## 이거 추가
        super(VitModel,self).__init__()
        super(torch.nn.Module,self).__init__()
        VitModel.__init__(self,
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads)
        if device is not None and dtype is not None:
            super().to(device, dtype=dtype)
            self.vit=self.vit.to(device, dtype=dtype)
            self.linear_projection=self.linear_projection.to(device, dtype=dtype)
            # self.flatten = torch.nn.Flatten().to(device, dtype=dtype)
            self.avgpool= torch.nn.AdaptiveAvgPool1d(1)
            patch_num=(image_size//patch_size)**2
            # self.classLayer = torch.nn.Linear(dim*patch_num, num_class).to(device, dtype=dtype)
            self.classLayer = torch.nn.Linear(patch_num, num_class).to(device, dtype=dtype)
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            # self.flatten = torch.nn.Flatten()
            self.avgpool= torch.nn.AdaptiveAvgPool1d(1)
            patch_num=(image_size//patch_size)**2
            self.classLayer = torch.nn.Linear(patch_num, num_class)
            # self.classLayer = torch.nn.Linear(dim*patch_num, num_class)
            self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid() 
    def forward(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                "Input image must have the shape [*, 3, {}, {}]".format(
                    self.image_size, self.image_size
                )
            )
        img_embeds = self.vit(img, return_embeddings=True)
        # img_embeds = self.flatten(img_embeds)
        img_embeds = self.avgpool(img_embeds)
        img_embeds = img_embeds.squeeze(-1)
        img_embeds=self.classLayer(img_embeds)
        img_embeds = self.sigmoid(img_embeds)
        return img_embeds