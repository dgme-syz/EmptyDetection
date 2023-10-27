import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils import data
from model import ResNet18


save_dir = os.path.join(Path(__file__).parent.absolute().__str__(), "Trained")
 
class MetricsAccumulator:
    def __init__(self):
        self.samples = 0
        self.loss = 0.0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def add_batch(self, samples, batch_loss, batch_tp, batch_fp, batch_tn, batch_fn):
        self.samples += samples
        self.loss += batch_loss
        self.tp += batch_tp
        self.fp += batch_fp
        self.tn += batch_tn
        self.fn += batch_fn
    
    def reset(self):
        self.samples = 0
        self.loss = 0.0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def calculate_loss(self):
        return self.loss / (self.samples + 1e-6)
    
    def calculate_accuracy(self):
        return (self.tp + self.tn) / (self.samples + 1e-6)
    
    def calculate_recall(self):
        return self.tp / (self.tp + self.fn + 1e-6)
    
    def calculate_precision(self):
        return self.tp / (self.tp + self.fp + 1e-6)
    
    def calculate_f1_score(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return 2 * precision * recall / (precision + recall + 1e-6)

def evaluate_model(net, data_iter, loss):
    accumulator = MetricsAccumulator()
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            l = loss(y_hat, y)# CrossEntropy输入 logits,labels
            preds = torch.argmax(y_hat, dim=1)
            tp = torch.sum((preds == y) & (y == 1))
            fp = torch.sum((preds != y) & (y == 0))
            tn = torch.sum((preds == y) & (y == 0))
            fn = torch.sum((preds != y) & (y == 1))
            accumulator.add_batch(y.numel(), l.sum().item(), tp.item(), fp.item(), tn.item(), fn.item())
    # print(accumulator.samples)
    # print((accumulator.tp + accumulator.tn) / (accumulator.tp + accumulator.tn + accumulator.fp + accumulator.fn  + 1e-6), ' -> ', accumulator.calculate_accuracy())
    metrics = {
        "loss": accumulator.calculate_loss(),
        "accuracy": accumulator.calculate_accuracy(),
        "recall": accumulator.calculate_recall(),
        "precision": accumulator.calculate_precision(),
        "f1_score": accumulator.calculate_f1_score()
    }
    accumulator.reset()
    return metrics

def save_model(net, epoch, acc, acc_best):
    """epoch & acc : 准确率 """
    torch.save(net.state_dict(), os.path.join(save_dir, f"epoch_{epoch}_acc_{acc:.6f}.pt"))
    if acc > acc_best:
        acc_best = acc
        torch.save(net.state_dict(), os.path.join(save_dir, f"best_model.pt"))

def resume(net):
    if "best_model.pt" in os.listdir(save_dir):
        net.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
        print("load success!")
    else:
        print("please get pretrained model")

if __name__ == '__main__':
    cnt = 100
    X = torch.normal(0, 1, (cnt, 3, 10, 10))
    y = torch.normal(0.5, 1, (cnt,))
    y = torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y)).long()
    print(X.shape)
    print(y.shape)
    dataset = data.TensorDataset(X, y)
    loss = nn.CrossEntropyLoss(reduction='none')
    net = ResNet18(num_classes=10)
    data_iter = data.DataLoader(dataset, batch_size=2, shuffle=False)
    metrics = evaluate_model(net, data_iter, loss)
    print(f"loss:{metrics['loss']} accuracy:{metrics['accuracy']}\
          recall:{metrics['recall']} precision:{metrics['precision']} f1_score:{metrics['f1_score']}")
    
    print(os.listdir(save_dir))
    
