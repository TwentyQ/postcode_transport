import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pth')


class ConvNeXtRegressor(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        return torch.sigmoid(self.base(x))


def build_model():
    base = models.convnext_tiny(weights=None)
    base.classifier[2] = nn.Linear(base.classifier[2].in_features, 1)
    return ConvNeXtRegressor(base)


valid_tf = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(image_path, weights_path=None):
    if weights_path is None:
        weights_path = _DEFAULT_WEIGHTS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.half()          # ← добавь эту строку
    model = model.to(device)
    model.eval()

    img = Image.open(image_path).convert('RGB')
    inp = valid_tf(img).unsqueeze(0).to(device)
    inp = inp.half()              # ← и эту

    with torch.no_grad():
        p1 = model(inp).item()
        p2 = model(torch.flip(inp, dims=[3])).item()
    pct = (p1 + p2) / 2 * 100

    return min(max(pct, 0.0), 100.0)