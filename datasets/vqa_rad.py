import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter

class VQARADDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        vocab=None,
        answer_vocab=None,
        transform=None,
        max_question_length=30,
        seed=42,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.max_question_length = max_question_length

        with open(os.path.join(data_dir, "VQA_RAD Dataset Public.json")) as f:
            data = json.load(f)

        # remove duplicates
        seen = set()
        unique = []
        for d in data:
            key = (d["image_name"], d["question"], d["answer"])
            if key not in seen:
                seen.add(key)
                unique.append(d)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(unique))
        n_train = int(0.7 * len(idx))
        n_val = int(0.8 * len(idx))

        if split == "train":
            self.data = [unique[i] for i in idx[:n_train]]
        elif split == "val":
            self.data = [unique[i] for i in idx[n_train:n_val]]
        else:
            self.data = [unique[i] for i in idx[n_val:]]

        self.vocab = vocab or self._build_vocab()
        self.answer_vocab = answer_vocab or self._build_answer_vocab()

    def _build_vocab(self):
        counter = Counter()
        for d in self.data:
            counter.update(d["question"].lower().split())
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for w in counter:
            vocab[w] = len(vocab)
        return vocab

    def _build_answer_vocab(self):
        answers = sorted({str(d["answer"]).lower().strip() for d in self.data})
        return {a: i for i, a in enumerate(answers)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        img_path = os.path.join(
            self.data_dir, "VQA_RAD Image Folder", d["image_name"]
        )
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = d["question"].lower().split()
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        ids = ids[: self.max_question_length]
        ids += [0] * (self.max_question_length - len(ids))

        answer = str(d["answer"]).lower().strip()

        return {
            "image": image,
            "question_ids": torch.tensor(ids),
            "question_length": torch.tensor(len(tokens)),
            "question_text": d["question"],
            "answer_id": torch.tensor(self.answer_vocab[answer]),
        }
