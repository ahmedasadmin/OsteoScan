from typing import List, Tuple
import torch
from torch import nn
from utils import * 
from data_loader import * 
from torch.utils.data import DataLoader
import torch.optim as optim 
from torchvision import models
from tqdm import tqdm
from resent import KneeResNet

def train(model, criterion, optimizer, train_loader, val_loader, scheduler, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        # ===== Training =====
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for X, y in tqdm(train_loader, ascii=" #", total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Train"):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            corrects += (preds == y).sum().item()
            total += y.size(0)

        scheduler(optimizer, epoch)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = corrects / total

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for X_val, y_val in tqdm(val_loader, ascii=" #", total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Val"):
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                outputs = model(X_val)
                loss = criterion(outputs, y_val)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_corrects += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_corrects / val_total

        print(f"Epoch {epoch + 1}: "
              f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f} | "
              f"Val Loss={val_epoch_loss:.4f}, Val Acc={val_epoch_acc:.4f}")

    return model

if __name__=="__main__":
    scanner_obj = ImageFileScanner(root_dir=config.DATA_DIR,
                                   extensions=(".png",".jpeg", ".jpg"))
    paths, labels = scanner_obj.scan()
    labels, class_to_indx = encode_labels(labels=labels)
    X_train, X_test, y_train, y_test = DatasetSpliter.split(paths, labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=222)

    reader = OpenCVColorReader()
    transform = KneeXrayTransform(size=(224, 224), normalize=True)
    train_dataset = KneeXrayDataset(X_train, y_train, reader, transform=transform)
    val_dataset   = KneeXrayDataset(X_val, y_val, reader, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KneeResNet(len(class_to_indx), freeze_backbone=False, pretrained = True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.backbone.parameters(), lr=1e-4)

    model = train(model, criterion, optimizer, train_loader,
                val_loader, scheduler=exp_lr_scheduler,
                num_epochs=100, device=device)




    import time 
    torch.save(model.state_dict(), (f'resnet18{time.strftime("%Y%m%d-%H%M%S")}.pt'))
    # for batch_indx, (image, label) in enumerate(train_loader): 
    #     print(f"[DEBUG] Batch Index: {batch_indx}, Image shape: {image.shape}, Image type: {image.dtype}Image lable:{label}")
    #     print(f"[DEBUG] Image:{image}")

    #     if batch_indx==2:
    #         break

