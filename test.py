import torch 
from data_loader import * 
from utils  import * 
from torch.utils.data import DataLoader
from torch import nn
from resent import KneeResNet
import torch.optim as optim
import argparse
import cv2 as cv
def test(model, model_path, test_loader, loss):
    """Test the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    running_loss = 0.0
    all_preds = [] 
    all_labels = [] 

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y= y.to(device)
            outputs = model(X)
            l = loss(outputs, y)
            _, preds = torch.max(outputs, 1)
            running_loss += l.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())


    return all_preds, all_labels 

if __name__=="__main__":
    device = torch.device ("cuda" if torch.cuda.is_available() else"cpu")
    parser  = argparse.ArgumentParser(description="Test the model on image or file based on argument passed")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
                        "--data-dir",
                        type = str, 
                        help = "Path to dataset folder"
    )
    group.add_argument(
        "--image-path",
        type = str,
        help="Path to the image-path"
    )   
    parser.add_argument(
        "--model-path",
        type=str,
        default="resnet1820260214-102747.pt",
        help="Path to trained model"
    )


    args = parser.parse_args()
    model = KneeResNet(config.NUM_CLASSES, freeze_backbone=False, pretrained=True)
    loss = nn.CrossEntropyLoss(reduction="sum")
    model_path:str = args.model_path if args.model_path else "resnet1820260214-102747.pt"
    if args.data_dir:
        root_dir = args.data_dir
        scanner_obj = ImageFileScanner(root_dir=root_dir,
                                        extensions=(".png",".jpeg", ".jpg"))
        paths, labels = scanner_obj.scan()
        labels, class_to_indx = encode_labels(labels=labels)
        X_train, X_test, y_train, y_test = DatasetSpliter.split(paths, labels)


        reader = OpenCVColorReader()
        transform = KneeXrayTransform(size=(224, 224), normalize=True)

        val_dataset   = KneeXrayDataset(X_test, y_test, reader, transform=transform)
        val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        all_preds, labels = test(model, model_path , test_loader=val_loader, loss=loss)
        classes_names = list (class_to_indx.keys()) 
        metrics_evaluation(all_preds, labels, classes=classes_names)
        plot_confusion_matrix(all_preds, labels, classes=classes_names)
        roc(model=model, loader=val_loader, device="cuda", num_classes=5)
    elif args.image_path:
        reader = OpenCVColorReader()
        transform = KneeXrayTransform(size=(244, 244), normalize=True)
        image = reader.read(args.image_path)
        original = cv.imread(args.image_path)
        original = cv.resize(original, (720, 480))
        image = transform(image).unsqueeze(0).to(device)
        # print(f"image shape: {image.shape}")
        with torch.no_grad():
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            model.eval()
            outputs = model(image)
            pred_class = torch.argmax(outputs, dim=1).item()
            cv.putText(original, f"class: {pred_class}", (20, 30), cv.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow("X-ray Image", original)
            cv.waitKey(0)
            cv.destroyAllWindows()



