import os
import torch
from PIL import Image
from pathlib import Path
from model import ResNet18
from datasets.data_utils import preprocess

base_dir = Path(__file__).parent.absolute().__str__()
save_dir = os.path.join(base_dir, "Trained")
net = ResNet18(num_classes=2)

assert "best_model.pt" in os.listdir(save_dir)
net.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
net.eval()

def dectect(img):
    out = preprocess(Image.open(img)).unsqueeze(0)
    with torch.no_grad():
        out = net(out)
    res = torch.argmax(out, dim=1).item()
    print(res)
    return res

if __name__ == '__main__':
    dectect(os.path.join(base_dir, "datasets", "train_paking", "0", "spot65.jpg"))