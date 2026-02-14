from typing import List, Tuple
from config import Config as config
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
import cv2 as cv
import torch 
import os 


def exp_lr_scheduler(optimzer, epoch, init_lr=config.BASE_LR, lr_decay_epoch = config.EPOCH_DECAY):
    """Decay learning rate by factor of DECAY_WEIGHT every lr_decay_epoch epochs"""
    lr = init_lr * (config.DECAY_WEIGHT**(epoch//lr_decay_epoch))

    if epoch % lr_decay_epoch:
        print(f"LR: is set to {lr}")
    
    for param_group in optimzer.param_groups:
        param_group['lr'] = lr 
    
    return optimzer


def metrics_evaluation(predictions, labels, classes):
    """Classification report"""
    print("[INFO] Classification Report:")
    print(
        classification_report(
            labels,          # y_true
            predictions,     # y_pred
            target_names=classes
        )
    )


def roc(model, loader, device, num_classes):
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import auc, roc_curve

    all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.4f}")

    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC")
    plt.legend()
    plt.savefig("roc curve.png")
    plt.show()



def plot_confusion_matrix(predictions, labels, classes):
    """Plot confusion matrix"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, predictions)  # y_true first

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion matrix.png")
    plt.show()


class DatasetSpliter:
    @staticmethod
    def split(X, y, test_size=0.2, seed=4444):
        return train_test_split(
            X, y, test_size=test_size, random_state=seed
     
        )


class ImageReader:
    def read(self, path:str):
        raise NotImplementedError
    
    
class OpenCVColorReader(ImageReader):
    def read(self, path: str):
        import cv2 as cv
        image = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)

class OpenCVGrayReader(ImageReader):
    def read(self, path:str):
        return cv.imread(path, cv.IMREAD_GRAYSCALE)
    
class ImageFileScanner:
    def __init__(self, root_dir:str, extensions: Tuple[str, ...]):
        self.root_dir = root_dir 
        self.extensions = extensions
    def scan(self)->Tuple[List[str], List[str]]:
        image_paths = []
        labels = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            for class_name in os.listdir(folder_path):
                class_path = os.path.join(folder_path, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                for idx, img in enumerate(os.listdir(class_path)):
                    if img.lower().endswith((".png", ".jpeg", "jpg")):
                        image_paths.append(os.path.join(class_path, img))
                        labels.append(class_name)
            print("[DEBUG] Number of classes", set(labels))
            return image_paths, labels 


def encode_labels(labels):
    classes = sorted(set(labels))
    class_to_index = {c: i for i, c in enumerate(classes)}
    encoded = [class_to_index[l] for l in labels]
    return encoded, class_to_index
