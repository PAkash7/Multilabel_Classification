# Aimonk Multilabel Classification

This repository contains code to train a deep learning model for multilabel classification on images, where the labels are annotated with 4 attributes, taking missing (`NA`) values into account.

## Use Cases & Context

Multilabel classification is distinct from multiclass classification because an image can have **multiple labels simultaneously**. In this specific scenario, an image might belong to Attribute 1 and Attribute 4, but not Attribute 2.

This architecture is extremely useful for real-world scenarios such as:

1. **E-commerce & Retail Tagging**: An image of a shirt could be simultaneously tagged with multiple attributes like `[Red, V-Neck, Cotton, Men's]`.
2. **Medical Imaging**: An X-Ray could have multiple conditions present at once, such as `[Pneumonia, Enlarged Heart, Rib Fracture]`.
3. **Content Moderation**: Automatically flagging a user-uploaded image for multiple policies concurrently, e.g., `[Violence, Nudity, Spoof]`.
4. **Autonomous Driving**: Dashcam object tagging where a frame needs to be flagged for the presence of `[Pedestrian, Stop Sign, Red Light, Bicyclist]`.

This implementation specifically excels in scenarios where data is imperfect—if you have millions of images but some human labelers skipped answering whether "Attribute 2" was present in an image, this model uses **Loss Masking** to still learn from the other 3 attributes without throwing the valuable image away.

## Requirements

- Python 3.8+
- PyTorch & Torchvision
- Pillow
- Numpy
- Matplotlib

Install the dependencies:

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Running the Code

### 1. Training

To train the model, you need a directory of images and the `labels.txt` file.

```bash
python train.py --data_dir /path/to/images --label_file /path/to/labels.txt --epochs 10
```

This will:

- Load weights from a pretrained ResNet-50.
- Handle missing `NA` labels properly via Loss Masking without deleting images.
- Handle class imbalance automatically by balancing PosWeights for BCE Loss.
- Train the model and plot the loss curve into `loss_curve.png`.
- Output the saved weights in `model_weights.pth`.

### 2. Inference

To infer the attributes for a given image using a trained model:

```bash
python inference.py /path/to/test_image.jpg --model_weights model_weights.pth
```

This will output the presence of each attribute (`1` or `0`) alongside their underlying predicted probabilities.

## Approach & Techniques

### Handling Missing Labels ('NA')

Instead of dropping images containing at least one missing attribute (which wastes valuable labeled information for the other attributes in that image), we implement **Loss Masking**.
In `train.py`, `masked_bce_loss` calculates Binary Cross Entropy component-wise. The loss elements corresponding to targets with missing information (`NaN`) are zero-out prior to scaling and backpropagation.

### Skewed/Imbalanced Dataset Problem

To tackle attribute imbalance (many more instances with Attribute A present vs Attribute B), the dataset automatically iterates over valid rows to determine label frequencies.
We instantiate PyTorch's `pos_weight` in `BCEWithLogitsLoss`:
\[ pos\_weight = \frac{number\_of\_negative\_samples}{number\_of\_positive\_samples} \]
This scales the CE gradients appropriately.

### Pre-processing and Augmentations

The current code scales images to 256x256 and takes a 224x224 Random Crop, along with Random Horizontal Flips before Normalization using ImageNet Means and Stds.

#### Additional Thoughts & Un-implemented Techniques Due to Time

- **Advanced Augmentation Rules**: `CutMix`, `MixUp`, `RandAugment`, or `ColorJitter` can significantly reduce overfitting in imbalanced domains. MixUp, however, would require slight modification to the masked loss logic to blend the uncertainties gracefully.
- **Focal Loss**: For high imbalance scenarios, Focal Loss provides a dynamic penalty focused intensely on hard-to-classify samples, instead of just a static class-weight multiplier (BCE `pos_weight`).
- **Data Stratification during Splitting**: Re-writing `MultilabelDataset` splits with `iterative-stratification` ensures that rare classes are split proportionally between train and val datasets. Missing out on stratification might lead to 0 samples evaluated during val for a severely minority class.
- **Early Stopping & Metrics**: Validation BCE Loss alone does not perfectly measure performance in Multilabel classification. It would be optimal to also aggregate and monitor the mean average precision (mAP) or weighted F1 Score per attribute, applying Early Stopping based on the highest mAP over successive epochs.
