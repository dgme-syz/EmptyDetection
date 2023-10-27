from tqdm import tqdm
from os.path import join
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.data_utils import DatasetFromFolder
from torch.utils.data.dataset import random_split
from torch.optim import Adam


from model import ResNet18
from tool_utils import MetricsAccumulator, evaluate_model, save_model, resume

acc_best = 0

def train_epoch(net, train_iter, loss, updater):
    net.train()
    total_loss = 0
    num_batches = len(train_iter)
    accumulator = MetricsAccumulator()
    with tqdm(total=num_batches, desc='Epoch') as pbar:
        for i, (X, y) in enumerate(train_iter):
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            total_loss += l.mean().item()
            pbar.update(1)
            pbar.set_postfix({'Loss': total_loss / (i+1)})
            ### add metrics
            preds = torch.argmax(y_hat, dim=1)
            tp = torch.sum((preds == y) & (y == 1))
            fp = torch.sum((preds != y) & (y == 0))
            tn = torch.sum((preds == y) & (y == 0))
            fn = torch.sum((preds != y) & (y == 1))
            accumulator.add_batch(y.numel(), l.sum().item(), tp.item(), fp.item(), tn.item(), fn.item())
            ###
    metrics = {
        "loss": accumulator.calculate_loss(),
        "accuracy": accumulator.calculate_accuracy(),
        "recall": accumulator.calculate_recall(),
        "precision": accumulator.calculate_precision(),
        "f1_score": accumulator.calculate_f1_score()
    }
    accumulator.reset()
    print(f"tra: loss:{metrics['loss']:.6f} acc:{metrics['accuracy']:.6f}\
          rec:{metrics['recall']:.6f} prec:{metrics['precision']:.6f} f1:{metrics['f1_score']:.6f}")
     
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, updater)
        metrics = evaluate_model(net, test_iter, loss)
        if (epoch + 1) % 5 == 0:
            save_model(net, epoch+1, metrics['accuracy'], acc_best)
        print(f"val: loss:{metrics['loss']:.6f} acc:{metrics['accuracy']:.6f}\
          rec:{metrics['recall']:.6f} prec:{metrics['precision']:.6f} f1:{metrics['f1_score']:.6f}")


if __name__ == '__main__':
    BaseFold = Path(__file__).parent.absolute().__str__()
    dataset = DatasetFromFolder(join(BaseFold, "datasets", "train_paking"))

    train_data, test_data = random_split(dataset, [0.8, 0.2])
    train_iter = DataLoader(train_data, batch_size=32, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=32, shuffle=False)
    
    print(len(train_iter))

    net = ResNet18(num_classes=2)
    resume(net)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = Adam(lr=0.001, params=net.parameters())
    num_epochs = 10

    train(net, train_iter, test_iter, loss, num_epochs, optimizer)