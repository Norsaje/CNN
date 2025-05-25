# app/model_pytorch.py

import torch
import timm
from pathlib import Path
from typing import List

class ResNet50Classifier:
    def __init__(
        self,
        weights_path: str = "best_resnet50.pth",
        device: str = "cuda",
    ):
        """
        Простая ResNet50 на одном кадре: для теста real-time пайплайна.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Обычный ResNet50
        self.model = timm.create_model(
            "resnet50",         # вместо tsm_resnet50
            pretrained=False,
            num_classes=33,
        )
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, img_tensor: torch.Tensor) -> str:
        """
        img_tensor: (1, C, H, W)
        """
        img_tensor = img_tensor.to(self.device)
        out = self.model(img_tensor)
        idx = out.argmax(dim=1).item()
        return chr(ord("А") + idx)

    @staticmethod
    def preprocess_frame(frame, transform) -> torch.Tensor:
        """
        frame: BGR numpy (H,W,3)
        transform: torchvision.transforms.Compose
        Возвращает: (1, C, H, W)
        """
        import cv2
        import torchvision.transforms as T

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(img)
        return img.unsqueeze(0)
