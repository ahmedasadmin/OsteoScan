# Knee X-Ray Classification with ResNet34

A deep learning project for classifying knee X-ray images into multiple categories using a fine-tuned ResNet34 architecture with PyTorch.

## Overview

This project implements a convolutional neural network (CNN) based on ResNet34 for automated knee X-ray image classification. The model can classify knee X-rays into 5 different categories and provides comprehensive evaluation metrics including confusion matrices and ROC curves.

## Features

- **Transfer Learning**: Utilizes pre-trained ResNet34 backbone for improved performance
- **Custom Classifier Head**: Multi-layer fully connected network with dropout for regularization
- **Flexible Training**: Configurable learning rate decay and batch size
- **Comprehensive Evaluation**: 
  - Classification reports with precision, recall, and F1-score
  - Confusion matrix visualization
  - Multi-class ROC curves with AUC scores
- **Dual Inference Modes**: Test on entire datasets or individual images
- **GPU Support**: CUDA-enabled training and inference


Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:

```
data/
├── fold_1/
│   ├── class_0/
│   │   ├── image1.png
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   └── class_n/
│       └── ...
└── fold_2/
    └── ...
```

Supported image formats: `.png`, `.jpg`, `.jpeg`

## Model Architecture

The `KneeResNet` class extends ResNet34 with a custom classifier:

- **Backbone**: ResNet34 (pre-trained on ImageNet)
- **Classifier Head**:
  - Linear(512 → 1024) + ReLU + BatchNorm + Dropout(0.5)
  - Linear(1024 → 512) + ReLU + Dropout(0.3)
  - Linear(512 → num_classes)

### Parameters:
- `num_classes`: Number of output classes
- `freeze_backbone`: Whether to freeze backbone weights (default: True)
- `pretrained`: Use pre-trained weights (default: True)

## Usage

### Testing on a Dataset

Evaluate the model on a test dataset and generate comprehensive metrics:

```bash
python test.py --data-dir /path/to/dataset --model-path resnet1820260214-102747.pt
```

This will:
- Split the dataset (80% train, 20% test)
- Generate classification report
- Plot confusion matrix
- Generate multi-class ROC curves

### Testing on a Single Image

Classify a single knee X-ray image:

```bash
python test.py --image-path /path/to/image.png --model-path resnet1820260214-102747.pt
```

This will:
- Load and preprocess the image
- Display the predicted class on the image
- Show the result in an OpenCV window

### Command Line Arguments

- `--data-dir`: Path to dataset folder (mutually exclusive with `--image-path`)
- `--image-path`: Path to single image for inference (mutually exclusive with `--data-dir`)
- `--model-path`: Path to trained model weights (default: `resnet1820260214-102747.pt`)


### Evaluation Metrics

- **`metrics_evaluation()`**: Prints detailed classification report
- **`plot_confusion_matrix()`**: Generates and saves confusion matrix heatmap
- **`roc()`**: Generates multi-class ROC curves with AUC scores

### Training Utilities

- **`exp_lr_scheduler()`**: Exponential learning rate decay scheduler

## Outputs

When running evaluation, the following outputs are generated:

1. **Console Output**: Classification report with precision, recall, F1-score
2. **`confusion_matrix.png`**: Heatmap visualization of prediction accuracy
3. **`roc_curve.png`**: Multi-class ROC curves with AUC scores



## GPU Support

The code automatically detects and uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To force CPU usage, modify the device selection in the scripts.

## Image Preprocessing

Images are automatically:
- Resized to 224×224 pixels (or 244×244 for single image inference)
- Converted to RGB format
- Normalized (when `normalize=True` in transform)

## Notes

- **Image Size Discrepancy**: There's a minor inconsistency in the test script where dataset images use (224, 224) but single image inference uses (244, 244). Consider standardizing to (224, 224).
- **Model Saving**: Ensure your trained model is saved with the correct path
- **Memory**: Adjust `BATCH_SIZE` based on available GPU memory
- **Class Names**: The model outputs integer class indices. You may want to maintain a mapping to meaningful class names.

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE` in config.py
- Use a smaller model or reduce classifier layer sizes

### CUDA Not Available
- Ensure PyTorch is installed with CUDA support
- Check CUDA driver compatibility
- Fall back to CPU by setting `device = "cpu"`

### No Images Found
- Verify dataset structure matches the expected format
- Check file extensions (.png, .jpg, .jpeg)
- Ensure proper read permissions

## License
MIT license

## Citation
If you use this code in your research, please cite:
@software{knee_xray_classification,
  author = {[Ahmed M. Abdelgaber]},
  title = {Detect and Asses Knee Osteoarthritis (OA)},
  year = {2026},
  url = {https://github.com/ahmedasadmmin/OsteoScan}
}

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information here]