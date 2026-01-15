# Med-VQA
Build and evaluate deep learning models that can answer questions about medical images

# Medical Visual Question Answering with CNN and Transformer VLM

This repository implements and compares:
- A CNN-based baseline (ResNet-50 + BiLSTM)
- A transformer-based visual–language model (BiomedCLIP + Q-Former + T5)

Experiments are conducted on the **VQA-RAD** dataset.

## Models

### CNN Baseline
- ResNet-50 visual encoder
- BiLSTM question encoder
- Late fusion + classification head
- Cross-entropy loss

### Transformer VLM
- Frozen BiomedCLIP ViT image encoder
- Trainable Q-Former (query-based cross-attention)
- Frozen T5-based language encoder
- Classification head
- Adaptive Focal Loss for class imbalance

Only the adapter and classifier are trained.

## Dataset

**VQA-RAD**
- 315 medical images
- 2,244 question–answer pairs
- Closed-ended and open-ended questions

Dataset is **not included** in this repository.

Download from the original authors and place it as:


