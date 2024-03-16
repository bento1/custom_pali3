from pali3.main import Pali3
from transformers import PreTrainedModel, PretrainedConfig
import torch
# https://stackoverflow.com/questions/73948214/how-to-convert-a-pytorch-nn-module-into-a-huggingface-pretrainedmodel-object
class CustomPALI3Config(PretrainedConfig):
    model_type = "mymodel"
    def __init__(
        self,
        version = 1,
        model_name='',
        image_size=512,
        patch_size=256,
        dim=512,
        depth=12, 
        heads=32, 
        enc_num_tokens=32100,
        enc_max_seq_len=1024,
        dec_num_tokens=32100,
        dec_max_seq_len=1024,
        enc_depth=12,
        enc_heads=32,
        dec_depth=12,
        dec_heads=32,
        seq_len=1024,
        device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
        **kwargs,
    ):
        self.version = version
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.enc_num_tokens = enc_num_tokens
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_num_tokens = dec_num_tokens
        self.dec_max_seq_len = dec_max_seq_len
        self.enc_depth = enc_depth
        self.enc_heads = enc_heads
        self.dec_depth = dec_depth
        self.dec_heads = dec_heads
        self.seq_len = seq_len
        self.device = device
        super().__init__(**kwargs)

class CustomPALI3(PreTrainedModel):
    config_class = CustomPALI3Config

    def __init__(self, config:CustomPALI3Config):
        super().__init__(config)
        self.model = CustomPALI3_(
            model_name=config.model_name,
            image_size=config.image_size,
            patch_size=config.patch_size,
            dim=config.dim,depth=config.depth, 
            heads=config.heads, 
            enc_num_tokens=config.enc_num_tokens,
            enc_max_seq_len=config.enc_max_seq_len,
            dec_num_tokens=config.dec_num_tokens,
            dec_max_seq_len=config.dec_max_seq_len,
            enc_depth=config.enc_depth,
            enc_heads=config.enc_heads,
            dec_depth=config.dec_depth,
            dec_heads=config.dec_heads,
            seq_len=config.seq_len,
            device=config.device,
            )
        
    def forward(self, img, prompt, output, mask):
        return self.model(img, prompt, output, mask)

class CustomPALI3_(Pali3,torch.nn.Module):
    def __init__(self,model_name,image_size,patch_size,
        dim,depth, heads, 
        enc_num_tokens, enc_max_seq_len,
        dec_num_tokens, dec_max_seq_len,
        enc_depth, enc_heads, dec_depth, dec_heads,
        seq_len,
        device,
        ):
        # super().__init__() ## 이거 추가
        # super(torch.nn.Module, self).__init__() ## 이거 추가
        super(Pali3,self).__init__()
        super(torch.nn.Module,self).__init__()
        Pali3.__init__(self,
            model_name=model_name,
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            enc_num_tokens=enc_num_tokens,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_max_seq_len=dec_max_seq_len,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            seq_len=seq_len,)

        
        if device is not None:
            self.pali_model.to(device)
            self.vit_model.vit.to(device)
            self.vit_model.linear_projection.to(device)


    def forward(self, img, prompt, output, mask):
        img_embeds = self.vit_model.process(img)
        result = self.pali_model(
            prompt, output, mask=mask, src_prepend_embeds=img_embeds
        )
        return result

    def generate(self, image, prompt, mask=None, attn_mask=None, model_name=None):
        if model_name:
            self.model_name = model_name

        if not self.model_name:
            raise ValueError(
                "model_name must be specidfied either in the class constructor or in the generate method"
            )
        visual_tokens = self.vit_model.process(image)
        text_tokens = self.pali_model.encoder(prompt)
        img_embeds = torch.cat((visual_tokens, text_tokens), dim=1)
        seq_out_start = torch.zeros(1, 1).long()
        result = self.pali_model.generate(
            img_embeds, seq_out_start, self.seq_len, mask, attn_mask
        )
        result_text = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return result_text
    
    def generate_(self, text, mask=None, attn_mask=None, model_name=None):
        if model_name:
            self.model_name = model_name

        if not self.model_name:
            raise ValueError(
                "model_name must be specidfied either in the class constructor or in the generate method"
            )

        seq_out_start = torch.zeros(1, 1).long()
        result = self.pali_model.generate(
            text, seq_out_start, self.seq_len, mask, attn_mask
        )
        result_text = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return result_text