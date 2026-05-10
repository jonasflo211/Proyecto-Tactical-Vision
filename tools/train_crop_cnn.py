import argparse
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CropDataset(Dataset):
    def __init__(self, class_dirs, image_size=(64, 128), augment=False):
        self.samples = []
        self.image_size = image_size
        self.augment = augment
        for label, folder in enumerate(class_dirs):
            folder = Path(folder)
            if not folder.exists():
                raise FileNotFoundError(f"No existe: {folder}")
            for path in folder.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError("No se encontraron imagenes de entrenamiento.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize(self.image_size, Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()
        return x, torch.tensor(label, dtype=torch.long)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyCropCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparable(16, 24, stride=1),
            DepthwiseSeparable(24, 32, stride=2),
            DepthwiseSeparable(32, 48, stride=2),
            DepthwiseSeparable(48, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


def export_onnx(model, dummy, output):
    common_kwargs = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": 11,
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
    }

    try:
        torch.onnx.export(
            model,
            dummy,
            output.as_posix(),
            dynamo=True,
            **common_kwargs,
        )
        print("Export ONNX usando exporter moderno de PyTorch.")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "onnxscript":
            raise
        print("onnxscript no esta instalado; cambiando al exporter clasico.")

    torch.onnx.export(
        model,
        dummy,
        output.as_posix(),
        dynamo=False,
        **common_kwargs,
    )
    print("Export ONNX usando exporter clasico de PyTorch.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", default="dataset/positives")
    parser.add_argument("--negatives", default="dataset/negatives")
    parser.add_argument("--team-dirs", nargs=3, metavar=("LOCAL", "VISITANTE", "ARBITRO"))
    parser.add_argument("--output", default="dataset/player_validator.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.team_dirs:
        class_dirs = args.team_dirs
        class_names = ["local", "visitante", "arbitro"]
    else:
        class_dirs = [args.negatives, args.positives]
        class_names = ["no_jugador", "jugador"]

    dataset = CropDataset(class_dirs, augment=True)
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyCropCNN(num_classes=len(class_dirs)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = accuracy(model, val_loader, device)
        print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 128, 64)
    export_onnx(model, dummy, output)
    labels = output.with_suffix(".labels.txt")
    labels.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"ONNX guardado: {output}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
