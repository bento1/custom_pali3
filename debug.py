import torch
from PIL import Image
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
import gc
import torch
gc.collect()
torch.cuda.empty_cache()


import torch
from torch import nn
from pali3.ul2 import UL2, ViTransformerWrapper, Encoder


class PrependTokens(nn.Module):
    """

    # Initialize models
    vit_model = ViTModel()
    text_embedding = TextEmbedding("bert-base-uncased")

    # Initialize PrependVisualTokens
    prepend_visual_tokens = PrependVisualTokens(vit_model, text_embedding)

    # Process image and text
    img = torch.randn(1, 3, 256, 256)  # dummy image
    text = "This is a sample text"
    combined_tokens = prepend_visual_tokens.process(img, text)

    """

    def __init__(
        self,
        vit,
        text_embedding,
    ):
        super().__init__()
        self.vit = vit
        self.text_embedding = text_embedding

    def forward(self, x):
        visual_tokens = self.vit.process(x)
        text_tokens = self.text_embedding.process(x)
        combined_tokens = torch.cat((visual_tokens, text_tokens), dim=1)
        return combined_tokens


class VitModel(nn.Module):
    """
    VitModel is a wrapper around the ViT model from the PyTorch Image Models library.

    Args:
        image_size (int, optional): Size of the image. Defaults to 256.
        patch_size (int, optional): Size of the patch. Defaults to 32.
        dim (int, optional): Dimension of the model. Defaults to 512.
        depth (int, optional): Depth of the model. Defaults to 6.
        heads (int, optional): Number of heads in the model. Defaults to 8.

    Raises:
        ValueError: If the input image is None.
        ValueError: If the input image shape is not [*, 3, image_size, image_size].

    Examples:
    x = torch.randn(1, 3, 256, 256)
    model = VitModel()
    out = model.process(x)
    print(out)




    """

    def __init__(
        self, image_size=256, patch_size=32, dim=512, depth=6, heads=8, *args, **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.depth = depth
        self.heads = heads

        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(dim=dim, depth=depth, heads=heads),
        )
        # adaptive avg pool
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear_projection = nn.Linear(dim, dim)

    def process(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                "Input image must have the shape [*, 3, {}, {}]".format(
                    self.image_size, self.image_size
                )
            )
        vit_output = self.vit(img, return_embeddings=True)
        # missing pool add later
        # pooled_output = self.pool(vit_output)
        projected_output = self.linear_projection(vit_output)
        return projected_output


class Pali3:
    """
    Pali3 is a vit model with a transformer encoder and decoder.
    It is a wrapper around the UL2 model from the PyTorch Image Models library.


    Args:
        model_name (str, optional): Name of the model. Defaults to None.
        image_size (int, optional): Size of the image. Defaults to 256.
        patch_size (int, optional): Size of the patch. Defaults to 32.
        dim (int, optional): Dimension of the model. Defaults to 512.
        depth (int, optional): Depth of the model. Defaults to 6.
        heads (int, optional): Number of heads in the model. Defaults to 8.
        enc_num_tokens (int, optional): Number of tokens in the encoder. Defaults to 256.
        enc_max_seq_len (int, optional): Maximum sequence length in the encoder. Defaults to 1024.
        dec_num_tokens (int, optional): Number of tokens in the decoder. Defaults to 256.
        dec_max_seq_len (int, optional): Maximum sequence length in the decoder. Defaults to 1024.
        enc_depth (int, optional): Depth of the encoder. Defaults to 6.
        enc_heads (int, optional): Number of heads in the encoder. Defaults to 8.
        dec_depth (int, optional): Depth of the decoder. Defaults to 6.
        dec_heads (int, optional): Number of heads in the decoder. Defaults to 8.
        seq_len (int, optional): Length of the sequence. Defaults to 1024.

    Raises:
        ValueError: If the model name is None.
        ValueError: If the tokenizer is None.

    Examples:

    model = Pali3()
    img = torch.randn(1, 3, 256, 256)  # dummy image
    prompt = "This is a sample text"
    output = "This is a sample text"
    mask = None
    result = model.process(img, prompt, output, mask)

    """

    def __init__(
        self,
        model_name=None,
        image_size=256,
        patch_size=32,
        dim=512,
        depth=6,
        heads=8,
        enc_num_tokens=256,
        enc_max_seq_len=1024,
        dec_num_tokens=256,
        dec_max_seq_len=1024,
        enc_depth=6,
        enc_heads=8,
        dec_depth=6,
        dec_heads=8,
        seq_len=1024,
    ):
        self.model_name = model_name
        self.vit_model = VitModel(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
        )

        self.pali_model = UL2(
            dim=dim,
            enc_num_tokens=enc_num_tokens,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            dec_max_seq_len=dec_max_seq_len,
        )
    
    def process(self, img, prompt, output, mask):
        img_embeds = self.vit_model.process(img)

        result = self.pali_model(
            prompt, output, mask=mask, src_prepend_embeds=img_embeds
        )
        return result

    def generate(self, text, mask=None, attn_mask=None, model_name=None):
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



img=Image.open('test.png').convert('RGB').resize((512,512))
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
device=torch.device("cpu")
# device=torch.device("cpu")
img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(dim=0).to(device,dtype=torch.float32)
query="summarize this chart"
tonkens = tokenizer(
        query, max_length=512, padding="max_length", truncation=True,return_tensors="pt")
prompt=tonkens['input_ids'].to(device)
mask = tonkens['attention_mask'].numpy()
mask=mask==1
mask=torch.tensor(mask).to(device)
answer="This trace chart displays the \u770c's values over time, with green indicating normal values and light green indicating abnormal ones. The overshoot of the abnormal signal is approximately 57% longer than that of the normal signals. Moreover, the two signals are easily distinguishable. The amplitude of the abnormal signal is smaller than that of the normal signal."
output_text = tokenizer(
        answer, max_length=1024, padding="max_length", truncation=True,return_tensors="pt")['input_ids'].to(device)

output_text = torch.randint(0, 256, (1, 1024)).to(device)

model = Pali3(model_name='test',
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
        seq_len=1024,)
model.vit_model.linear_projection.to(device)
model.vit_model.vit.to(device)
model.pali_model.to(device)
result=model.process(img, prompt, output_text, mask)

