import torch
import torch.nn as nn
import open_clip
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .qformer import QFormer

class TransformerVLM(nn.Module):
    def __init__(self, num_answers, device):
        super().__init__()
        self.device = device

        self.vision, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        for p in self.vision.parameters():
            p.requires_grad = False

        self.qformer = QFormer()

        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
        self.text = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
        for p in self.text.parameters():
            p.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

        self.classifier = nn.Linear(768, num_answers)

    def forward(self, images, questions):
        with torch.no_grad():
            patches = self.vision.visual.trunk.patch_embed(images)
            patches = self.vision.visual.trunk.norm(patches)

        qv = self.qformer(patches)[:, 0]

        with torch.no_grad():
            toks = self.tokenizer(
                questions, return_tensors="pt",
                padding=True, truncation=True
            ).to(self.device)
            enc = self.text.encoder(**toks).last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)
            qt = (enc * mask).sum(1) / mask.sum(1)

        fused = self.fusion(torch.cat([qv, qt], dim=1))
        return self.classifier(fused)
