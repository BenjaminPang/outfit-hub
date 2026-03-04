import io
import numpy as np
from typing import List, Union

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class ClipEmbedding:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and move to GPU
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, inputs_list: List[Union[bytes, str]]) -> np.ndarray:
        """
        Extract image embeddings from either raw bytes or file paths.
        
        Args:
            inputs_list: A list containing either image bytes (from TAR) or 
                         string paths (from local disk).
        """
        images = []
        for item in inputs_list:
            if isinstance(item, str):
                # It's a file path
                img = Image.open(item).convert("RGB")
            else:
                # It's raw bytes
                img = Image.open(io.BytesIO(item)).convert("RGB")
            images.append(img)
        
        # Standard CLIP preprocessing
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Efficient vision-only inference
        image_embeds = self.model.get_image_features(**inputs)
        
        # L2 Normalization is essential for cosine similarity downstream
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return image_embeds.cpu().numpy()

    @torch.no_grad()
    def get_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract only text embeddings.
        """
        inputs = self.processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(self.device)
        
        text_embeds = self.model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return text_embeds.cpu().numpy()
