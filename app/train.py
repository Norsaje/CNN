#!/usr/bin/env python3
# app/train.py

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import decord
from decord import VideoReader, cpu

from sklearn.metrics import accuracy_score

# -----------------------------
# 1) Dataset для bukva
# -----------------------------
class BukvaVideoDataset(Dataset):
    def __init__(self, root: Path, annotations_file: Path,
                 frames_per_clip: int = 8, transform=None):
        """
        root: папка bukva/data
        annotations_file: bukva/annotations.tsv (формат: video_path \t label)
        """
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        # читаем аннотации
        lines = annotations_file.read_text().strip().splitlines()
        self.samples = []
        for line in lines[1:]:  # пропустить header
            vid_path, label = line.split('\t')[:2]
            self.samples.append((root/vid_path, int(label)))  # метки числами

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        # равномерно берём кадры
        indices = torch.linspace(0, total_frames - 1, self.frames_per_clip).long().tolist()
        frames = vr.get_batch(indices).asnumpy()  # форма (T, H, W, 3)

        # превратить в тензор (T,3,H,W) и нормализовать
        clip = []
        for img in frames:
            img = transforms.ToPILImage()(img)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            clip.append(img)
        clip = torch.stack(clip)  # (T, C, H, W)
        # для модели TSM ожидаем (C, T, H, W)
        clip = clip.permute(1, 0, 2, 3)
        return clip, label

# -----------------------------
# 2) Модель: TSM с ResNet50
# -----------------------------
def build_model(num_classes: int, pretrained: bool = True):
    # используем библиотеку timm с реализацией TSM
    model = timm.create_model(
        'resnet50_tsm',
        pretrained=pretrained,
        num_classes=num_classes,
        num_segments=8,         # фреймов на клип
        shift_div=8             # TSM hyperparam
    )
    return model

# -----------------------------
# 3) Цикл тренировки
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, targets = [], [], []
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        out = model(clips)           # (B, num_classes)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return sum(losses)/len(losses), acc

def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            out = model(clips)
            loss = criterion(out, labels)

            losses.append(loss.item())
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return sum(losses)/len(losses), acc

# -----------------------------
# 4) Main: аргументы и запуск
# -----------------------------
def main():
    p = ArgumentParser()
    p.add_argument('--bukva-root', type=Path, default=Path('bukva/data'))
    p.add_argument('--ann-file', type=Path, default=Path('bukva/annotations.tsv'))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    # Аугментации и нормализация
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # Датасеты
    full = BukvaVideoDataset(args.bukva_root, args.ann_file, transform=transform)
    # простая разбивка train/val 80/20
    val_size = int(0.2 * len(full))
    train_set, val_set = torch.utils.data.random_split(full, [len(full)-val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Модель
    num_classes = max(s[1] for s in full.samples) + 1
    model = build_model(num_classes).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc     = validate(model, val_loader,    criterion, args.device)
        scheduler.step()

        print(f"[Epoch {epoch:02d}] "
              f"Train: loss={train_loss:.3f}, acc={train_acc:.3f} | "
              f"Val:   loss={val_loss:.3f}, acc={val_acc:.3f}")

        # сохраняем лучший чекпоинт
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_tsm_resnet50.pth')
            print(f" → New best: {best_acc:.3f}, model saved.")

if __name__ == '__main__':
    main()
