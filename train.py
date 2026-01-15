import json
from datetime import datetime

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.vqa_rad import VQARADDataset
from models.cnn import CNNBaseline
from models.vlm import TransformerVLM
from losses.focal import AdaptiveFocalLoss
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint

def init_metrics_log(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)

def append_metrics(path, record):
    with open(path, "r") as f:
        data = json.load(f)
    data.append(record)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def train_one_epoch(model, loader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0.0

    for b in loader:
        optimizer.zero_grad()

        images = b["image"].to(device)
        labels = b["answer_id"].to(device)

        if model_type == "cnn":
            logits = model(
                images,
                b["question_ids"].to(device),
                b["question_length"].to(device),
            )
        else:
            logits = model(images, b["question_text"])

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, model_type):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for b in loader:
            images = b["image"].to(device)
            labels = b["answer_id"].to(device)

            if model_type == "cnn":
                logits = model(
                    images,
                    b["question_ids"].to(device),
                    b["question_length"].to(device),
                )
            else:
                logits = model(images, b["question_text"])

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # -------------------------
    # Dataset
    # -------------------------
    train_set = VQARADDataset(args.data_dir, split="train", transform=transform)
    val_set = VQARADDataset(
        args.data_dir,
        split="val",
        vocab=train_set.vocab,
        answer_vocab=train_set.answer_vocab,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=4
    )

    # -------------------------
    # Model
    # -------------------------
    if args.model == "cnn":
        model = CNNBaseline(
            vocab_size=len(train_set.vocab),
            num_answers=len(train_set.answer_vocab),
        )
        criterion = torch.nn.CrossEntropyLoss()
    else:
        model = TransformerVLM(
            num_answers=len(train_set.answer_vocab),
            device=device,
        )
        class_freq = torch.bincount(
            torch.tensor(
                [d["answer_id"] for d in train_set],
                dtype=torch.long,
            ),
            minlength=len(train_set.answer_vocab),
        ).float().to(device)
        criterion = AdaptiveFocalLoss(class_freq, gamma=1.0)

    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    # -------------------------
    # Checkpoint resume
    # -------------------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(args.ckpt_dir, "best_model.pt")
    last_ckpt = os.path.join(args.ckpt_dir, "last.pt")
    
    metrics_path = os.path.join(args.ckpt_dir, "metrics.json")
    init_metrics_log(metrics_path)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_acc = load_checkpoint(
            model,
            optimizer,
            args.resume,
            device=device,
        )

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.model
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, args.model
        )
        
        append_metrics(
            metrics_path,
            {
                "timestamp": datetime.utcnow().isoformat(),
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "best_val_accuracy": round(best_val_acc, 6),
            }
        )

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_ckpt,
                best_metric=best_val_acc,
            )

        # Always save last checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            last_ckpt,
            best_metric=best_val_acc,
        )

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vlm"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    main(args)
