import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms
from torchvision.utils import save_image
from backbones import get_backbone
from models import get_model
import os
from sklearn.metrics import roc_curve, auc
from utils.util_vis import draw_roc_curve

class EmbedNet(pl.LightningModule):
    def __init__(self, backbone, model):
        super(EmbedNet, self).__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, x):
        x = self.backbone(x)
        embedded_x = self.model(x)
        return embedded_x

class TripletNet(pl.LightningModule):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)

class LightningTripletNet(pl.LightningModule):
    def __init__(self, config):
        super(LightningTripletNet, self).__init__()
        self.config = config
        backbone = get_backbone(self.config.backbone)
        model = get_model(self.config.model)
        embed_net = EmbedNet(backbone, model)
        self.triplet_net = TripletNet(embed_net)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, a, p, n):
        return self.triplet_net(a, p, n)

    def training_step(self, batch, batch_idx):
        a, p, n = batch
        embedded_a, embedded_p, embedded_n = self.triplet_net(a, p, n)
        loss = nn.TripletMarginLoss(margin=self.config.margin)(embedded_a, embedded_p, embedded_n)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a, p, n = batch
        embedded_a, embedded_p, embedded_n = self.triplet_net(a, p, n)
        loss = nn.TripletMarginLoss(margin=self.config.margin, reduction='none')(embedded_a, embedded_p, embedded_n)
        dist_pos = F.pairwise_distance(embedded_a, embedded_p)
        dist_neg = F.pairwise_distance(embedded_a, embedded_n)
        self.validation_step_outputs.append((loss, dist_pos, dist_neg))
        return loss, dist_pos, dist_neg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def on_validation_epoch_end(self):
        loss = torch.cat([x for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        dist_pos = torch.cat([y for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        dist_neg = torch.cat([z for x, y, z in self.validation_step_outputs]).detach().cpu().numpy()
        avg_loss = np.mean(loss)
        avg_dist_pos = np.mean(dist_pos)
        avg_dist_neg = np.mean(dist_neg)
        self.validation_step_outputs.clear()
        self.log("val_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("dist_pos", avg_dist_pos, prog_bar=True, logger=True, sync_dist=True)
        self.log("dist_neg", avg_dist_neg, prog_bar=True, logger=True, sync_dist=True)

        y_true = np.concatenate([np.ones_like(dist_pos), np.zeros_like(dist_neg)])
        y_scores = np.concatenate([dist_pos, dist_neg])
        fpr, tpr, thresholds = roc_curve(y_true, -y_scores)
        roc_auc = auc(fpr, tpr)
        draw_roc_curve(fpr, tpr, save_path=f'roc_curve_epoch_{self.current_epoch}.png', roc_auc=roc_auc)


        return avg_loss, avg_dist_pos, avg_dist_neg

    def test_step(self, batch, batch_idx):
        a, p, n = batch
        embedded_a, embedded_p, embedded_n = self.triplet_net(a, p, n)
        loss = nn.TripletMarginLoss(margin=self.config.margin, reduction='none')(embedded_a, embedded_p, embedded_n)
        dist_pos = F.pairwise_distance(embedded_a, embedded_p)
        dist_neg = F.pairwise_distance(embedded_a, embedded_n)
        self.test_step_outputs.append((a, p, n, dist_pos, dist_neg))
        return loss

    def on_test_epoch_end(self):
        saved_count = 0
        dist_pos_list = []
        dist_neg_list = []
        
        for batch_idx, (a, p, n, dist_pos, dist_neg) in enumerate(self.test_step_outputs):
            dist_pos_list.append(dist_pos.cpu().numpy())
            dist_neg_list.append(dist_neg.cpu().numpy())
            
            for i in range(len(dist_pos)):
                if saved_count >= 10:
                    break
                if dist_pos[i] > self.config.margin:
                    self.save_images(a[i], p[i], n[i], batch_idx, i, dist_pos[i], 'pos')
                    saved_count += 1
                if dist_neg[i] < self.config.margin:
                    self.save_images(a[i], p[i], n[i], batch_idx, i, dist_neg[i], 'neg')
                    saved_count += 1
            if saved_count >= 10:
                break

        self.test_step_outputs.clear()

        dist_pos = np.concatenate(dist_pos_list)
        dist_neg = np.concatenate(dist_neg_list)
        
        y_true = np.concatenate([np.ones_like(dist_pos), np.zeros_like(dist_neg)])
        y_scores = np.concatenate([dist_pos, dist_neg])
        fpr, tpr, thresholds = roc_curve(y_true, -y_scores) 
        roc_auc = auc(fpr, tpr)
        draw_roc_curve(fpr, tpr, save_path='roc_curve_test.png', roc_auc=roc_auc)

    def save_images(self, anchor, positive, negative, batch_idx, img_idx, wrong, label_type):
        os.makedirs('misclassified', exist_ok=True)
        wrong_str = f"{wrong:.2f}"
        if label_type == 'pos':
            self._save_image(anchor, f'misclassified/{batch_idx}_{img_idx}_{wrong_str}_anchor_pos.png')
            self._save_image(positive, f'misclassified/{batch_idx}_{img_idx}_{wrong_str}_positive.png')
        elif label_type == 'neg':
            self._save_image(anchor, f'misclassified/{batch_idx}_{img_idx}_{wrong_str}_anchor_neg.png')
            self._save_image(negative, f'misclassified/{batch_idx}_{img_idx}_{wrong_str}_negative.png')
        

    def _save_image(self, tensor, filepath):
        inv_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.ToPILImage()
        ])
        inv_image = inv_transform(tensor)
        inv_image.save(filepath)