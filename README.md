# Medical Visual Question Answering (Med-VQA)

This repository implements and evaluates two models on the VQA-RAD dataset:

- CNN Baseline (ResNet-50 + BiLSTM)
- Transformer-based VLM (BiomedCLIP + Q-Former + T5)

The goal is to compare performance on **closed-ended vs open-ended** medical questions.

## Dataset

**VQA-RAD** (not included):

```
data/vqa_rad/
├── VQA_RAD Image Folder/
└── VQA_RAD Dataset Public.json
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

CNN:
```bash
python train.py --model cnn --data_dir ./data/vqa_rad
```

VLM:
```bash
python train.py --model vlm --data_dir ./data/vqa_rad
```

## Evaluation (Closed vs Open)

```bash
python evaluation.py   --model vlm   --data_dir ./data/vqa_rad   --ckpt path/to/best_model.pt
```

## Notes
- Research / educational use only
- Not for clinical deployment

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

Download from the original authors.
