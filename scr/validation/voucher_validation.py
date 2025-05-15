#scr/validation/voucher_validation.py
import torch
from typing import List
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class Validator:
    def __init__(self, checkpoint: str, device: torch.device, labels: List[str], confidence_threshold: float = 0.8):
        self.device = device
        self.model  = CLIPModel.from_pretrained(checkpoint).to(device)
        self.processor  = CLIPProcessor.from_pretrained(checkpoint)
        self.labels = labels
        self.threshold = confidence_threshold

    def is_voucher(self, image: Image.Image) -> bool:
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()
        try:
            idx = self.labels.index('voucher')
            prob = probs[idx]
        except ValueError:
            prob = max(probs)
        return prob >= self.threshold
