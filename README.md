# HVUE Architecture for Image Segmentation

## Background
Implementation of the Hybrid Vision U-Net Ensemble (HVUE) architecture adapted for forest cover segmentation from satellite imagery. The architecture combines U-Net with ResNet50 and Vision Transformer (ViT) feature extractors to achieve robust semantic segmentation.

## Architecture Components

### Base Architecture: HVUE
- **Source Paper**: "A novel hybrid vision UNet architecture for brain tumor segmentation and classification" (Nature Scientific Reports, 2025)
- **Paper Link**: https://www.nature.com/articles/s41598-025-09833-y#Sec8

### Key Modifications from Original Paper
1. **Feature Extractor Upgrade**: ResNet50 (replacing DenseNet121 from original paper)
   - Pretrained on ImageNet (IMAGENET1K_V2 weights)
   - Extracts 512-channel features at 16×16 resolution
   
2. **Vision Transformer**: ViT-Tiny (Patch 16)
   - Generates 3-channel feature maps at 16×16 resolution
   - Provides global context understanding

3. **U-Net Encoder-Decoder**:
   - Standard U-Net with 4 encoder and 4 decoder blocks
   - Skip connections preserve spatial information
   - Bottleneck fusion layer combines all feature sources

### Architecture Flow
```
Input (256×256 RGB) → U-Net Encoder (4 blocks)
                    ↓
                Bottleneck (1024 channels)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
ResNet50      Bottleneck        ViT-Tiny
(512 ch)      (1024 ch)         (3 ch)
    └───────────────┼───────────────┘
                    ↓
            Feature Fusion (1537→1024 channels)
                    ↓
            U-Net Decoder (4 blocks)
                    ↓
            Output (1 channel binary mask)
```

## Dataset

### Forest Segmentation Dataset
- **Source**: DeepGlobe 2018 Challenge - Land Cover Classification Track
- **Task**: Binary segmentation of forest regions in satellite imagery
- **Total Images**: 5,108 aerial images (256×256 pixels)
- **Format**: RGB images with corresponding binary masks
- **Split**: 70% training / 30% validation

### Citation
```bibtex
@InProceedings{DeepGlobe18,
  author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and 
            Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and 
            Tuia, Devis and Raskar, Ramesh},
  title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2018}
}
```

## Implementation Information

### Training Configuration
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Learning Rate Scheduler**: ReduceLROnPlateau (patience=5)
- **Mixed Precision**: Automatic Mixed Precision (AMP) enabled on GPU
- **Gradient Clipping**: Max norm = 1.0
- **Epochs**: 5 (configurable)

## Data Augmentation

### Training Transforms (Albumentations)
- Resize: 256×256
- Horizontal/Vertical Flip: p=0.5
- Random 90° Rotation: p=0.5
- ShiftScaleRotate: shift=0.0625, scale=0.1, rotate=45°
- Elastic/Grid/Optical Distortion: p=0.3
- ImageNet Normalization

### Validation Transforms
- Resize: 256×256
- ImageNet Normalization only

## Evaluation Metrics

The model reports comprehensive segmentation metrics:

1. **Dice Coefficient**: Overlap-based similarity metric
2. **IoU (Jaccard Index)**: Intersection over Union
3. **Pixel Accuracy**: Proportion of correctly classified pixels
4. **Precision**: True Positives / (TP + False Positives)
5. **Recall (Sensitivity)**: True Positives / (TP + False Negatives)
6. **F1 Score**: Harmonic mean of Precision and Recall
7. **Specificity**: True Negatives / (TN + False Positives)

## Model Output

### Saved Files
- `best_model_hvue.pth`: Best model checkpoint (lowest validation loss)
- `trained_model_final.pth`: Final model after all epochs

### Model Parameters
- **Total Parameters**: ~30M (exact count varies)
- **Trainable Parameters**: ~30M
- **Output**: Single-channel probability map (0-1 range after sigmoid)

## Usage Notes

1. **Memory Management**: The code includes aggressive memory management (garbage collection, cache clearing) to prevent OOM errors
2. **NaN Handling**: Automatic detection and skipping of batches with NaN values
3. **Checkpoint Saving**: Model automatically saves when validation loss improves
4. **Multi-threading**: Uses 2 workers for data loading with prefetching on GPU
