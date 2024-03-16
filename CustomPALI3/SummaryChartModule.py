from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math, os

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


class SummaryChartModule(pl.LightningModule):
    def __init__(self, tokenizer,args, model, train_dataset, val_dataset):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.validation_step_outputs=[]

    def training_step(self, batch, batch_idx):
        # pixel_values, decoder_input_ids, labels = batch
        # {'image':input_image_tensor ,"input_ids": input_batch, 'label':label_batch}
        newBatchSet=[]
        for b in batch['iamge']:
            img, prompt, output, mask = b
        for b in batch:
            img, prompt, output, mask = b
        for b in batch:
            img, prompt, output, mask = b
        
        logits, loss = self.model( img, prompt, output, mask)
        self.log_dict({"train_loss": loss}, sync_dist=True)
        print({"train_loss": loss})
        return loss
    
    def compute_metric(self,answer, pred):
        scores=sentence_bleu([answer.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25) )
        return 1/(scores+1e-3)
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        for b in batch:
            img, prompt = b

        
        outputs = self.model.generate(img, prompt)
    
        predictions = []
        for seq in self.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
            predictions.append(seq)
        
        scores = list()
        for pred, answer in zip(predictions, prompt):
            scores.append(self.compute_metric(answer, pred))
        self.validation_step_outputs.append(scores)

        return scores

    def on_validation_epoch_end(self):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            self.validation_step_outputs = [self.validation_step_outputs]
        assert len(self.validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        print("Epoch:", str(self.current_epoch), "Step:", str(self.global_step), "Validation Metric:", str(np.sum(total_metric) / np.sum(cnt)))
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        max_iter = None
        if int(self.args.get("max_epochs", -1)) > 0:
            assert len(self.args.get("train_batch_sizes")) == 1, "Set max_epochs only if the number of datasets is 1"
            max_iter = (self.args.get("max_epochs") * self.args.get("num_training_samples_per_epoch")) / (
                # self.args.get("train_batch_sizes")[0] * torch.cuda.device_count() * self.args.get("num_nodes", 1)
                self.args.get("train_batch_sizes")[0] * 1 * self.args.get("num_nodes", 1)
            )

        if int(self.args.get("max_steps", -1)) > 0:
            max_iter = min(self.args.get("max_steps"), max_iter) if max_iter is not None else self.args.get("max_steps")

        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.get("lr"))
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.args.get("warmup_steps")),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.get('train_batch_sizes')[0], shuffle=True, num_workers=self.args.get('num_workers'))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.get('valid_batch_size')[0], shuffle=False, num_workers=self.args.get('num_workers'))

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(self.args['output_dir'], 'summary_chart-checkpoint-epoch='+str(self.current_epoch)+'-'+str(self.global_step))
        self.model.save_pretrained(save_path, from_pt=True)
        self.processor.save_pretrained(save_path, from_pt=True)