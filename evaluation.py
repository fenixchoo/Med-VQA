import torch
import json
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

from datasets.vqa_rad import VQARADDataset
from models.cnn import CNNBaseline
from models.vlm import TransformerVLM

def evaluate(model, loader, device, model_type):
    model.eval()
    preds, labels, types = [], [], []

    with torch.no_grad():
        for b in loader:
            images = b["image"].to(device)
            y = b["answer_id"].to(device)

            if model_type == "cnn":
                out = model(
                    images,
                    b["question_ids"].to(device),
                    b["question_length"].to(device),
                )
            else:
                out = model(images, b["question_text"])

            preds.extend(out.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            types.extend(b["answer_type"])

    results = {}
    for t in ["OPEN", "CLOSED"]:
        idx = [i for i, v in enumerate(types) if v == t]
        if idx:
            acc = accuracy_score(
                [labels[i] for i in idx],
                [preds[i] for i in idx]
            )
            results[t] = acc

    results["OVERALL"] = accuracy_score(labels, preds)
    return results

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test = VQARADDataset(
        args.data_dir,
        split="test",
        transform=tfm
    )
    loader = DataLoader(test, batch_size=args.batch, shuffle=False)

    if args.model == "cnn":
        model = CNNBaseline(len(test.vocab), len(test.answer_vocab))
    else:
        model = TransformerVLM(len(test.answer_vocab), device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    model.to(device)

    metrics = evaluate(model, loader, device, args.model)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn","vlm"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=16)
    main(p.parse_args())