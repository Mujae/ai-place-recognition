import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import transforms
from backbones import get_backbone
from models import get_model
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils.util_vis import draw_roc_curve, draw_confusion_matrix, find_best_threshold
from setup import config
from PIL import Image

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
        best_threshold = find_best_threshold(fpr, tpr, thresholds)
        draw_roc_curve(fpr, tpr, thresholds, best_threshold=best_threshold, save_path=config.base_dir+f'/roc_curve_epoch_{self.current_epoch}.png', roc_auc=roc_auc)
        y_pred = (-y_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        draw_confusion_matrix(cm, best_threshold, save_path=config.base_dir+f'/confusion_matrix_epoch_{self.current_epoch}.png')

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
        dist_pos = torch.cat([a for x, y, z, a, r in self.test_step_outputs]).detach().cpu().numpy()
        dist_neg = torch.cat([r for x, y, z, a, r in self.test_step_outputs]).detach().cpu().numpy()

        y_true = np.concatenate([np.ones_like(dist_pos), np.zeros_like(dist_neg)])
        y_scores = np.concatenate([dist_pos, dist_neg])
        fpr, tpr, thresholds = roc_curve(y_true, -y_scores)
        roc_auc = auc(fpr, tpr)
        best_threshold = find_best_threshold(fpr, tpr, thresholds)
        draw_roc_curve(fpr, tpr, thresholds, best_threshold=best_threshold, save_path=config.base_dir+'/roc_curve_test.png', roc_auc=roc_auc)
        y_pred = (-y_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        draw_confusion_matrix(cm, best_threshold, save_path=config.base_dir+f'/confusion_matrix.png')

        for batch_idx, (a, p, n, dist_pos, dist_neg) in enumerate(self.test_step_outputs):
            for i in range(len(dist_pos)):
                if dist_pos[i] <= -best_threshold:
                    self.save_misclassified_images(a[i], p[i], n[i], batch_idx, i, dist_pos[i], 'pos_correct')

                if dist_neg[i] > -best_threshold:
                    self.save_misclassified_images(a[i], p[i], n[i], batch_idx, i, dist_neg[i], 'neg_correct')

    def save_misclassified_images(self, anchor, positive, negative, batch_idx, img_idx, score, label_type):
        os.makedirs('correct_predictions', exist_ok=True)
        score_str = f"{score:.2f}"
        file_prefix = f'correct_predictions/{batch_idx}_{img_idx}_{score_str}_'

        if label_type == 'pos_correct':
            combined_image = self.combine_images(anchor, positive)
            combined_image.save(f'{file_prefix}correct_pos.png')
        elif label_type == 'neg_correct':
            combined_image = self.combine_images(anchor, negative)
            combined_image.save(f'{file_prefix}correct_neg.png')

    def combine_images(self, anchor, second_image):
        """Anchor와 Positive 또는 Negative 이미지를 한 PNG 파일에 저장"""
        # GPU에 있을 수 있으니 CPU로 변환
        anchor = anchor.cpu()
        second_image = second_image.cpu()

        # 이미지를 PIL로 변환
        inv_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.ToPILImage()
        ])

        anchor_image = inv_transform(anchor)
        second_image_pil = inv_transform(second_image)

        # Anchor와 Positive 또는 Negative 이미지를 가로로 붙임
        total_width = anchor_image.width + second_image_pil.width
        max_height = max(anchor_image.height, second_image_pil.height)
        combined_image = Image.new('RGB', (total_width, max_height))

        # 좌우로 이미지를 붙임
        combined_image.paste(anchor_image, (0, 0))
        combined_image.paste(second_image_pil, (anchor_image.width, 0))

        return combined_image