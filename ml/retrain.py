import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class PhotoDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        img = Image.open(path).convert('RGB')
        label = torch.tensor(row['load_pct'] / 100.0)
        return self.transform(img), label


class ConvNeXtRegressor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        return torch.sigmoid(self.base(x))


def build_model(arch):
    if arch == 'convnext_tiny':
        base = models.convnext_tiny(weights=None)
        base.classifier[2] = nn.Linear(base.classifier[2].in_features, 1)
        return ConvNeXtRegressor(base)
    elif arch == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1), nn.Sigmoid())
        return m
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arch', default='convnext_tiny',
                        choices=['convnext_tiny', 'resnet18'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Новых фото: {len(df)}")

    train_tf = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loader = DataLoader(
        PhotoDataset(df, args.images, train_tf),
        batch_size=args.batch,
        shuffle=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Архитектура:", args.arch)

    model = build_model(args.arch)
    model.load_state_dict(torch.load(args.base, map_location=device))
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg = total_loss / len(df)
        print(f"Epoch {epoch+1}: MAE={avg*100:.2f}")

    torch.save(model.state_dict(), args.out)
    print(f"Новые веса сохранены: {args.out}")


if __name__ == '__main__':
    main()