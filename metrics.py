import torch
import torchmetrics

def calculate_metrics(preds, targets):
    preds = torch.argmax(preds, dim=1)
    targets = torch.argmax(targets, dim=1)
    print(preds.shape)
    print(targets.shape)
    accuracy = torchmetrics.Accuracy(num_classes=4, task="multiclass").cuda()
    precision = torchmetrics.Precision(num_classes=4, average="micro", task="multiclass").cuda()
    recall = torchmetrics.Recall(num_classes=4, average="micro", task='multiclass').cuda()
    f1_score = torchmetrics.F1Score(num_classes=4, average="micro", task='multiclass').cuda()
    print("f1_score")
    acc = accuracy(preds, targets)
    prec = precision(preds, targets)
    rec = recall(preds, targets)
    f1 = f1_score(preds, targets)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }