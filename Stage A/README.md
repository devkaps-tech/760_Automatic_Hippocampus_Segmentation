# Stage A: Multimodal 3D Attention U-Net for Hippocampus Segmentation and CDR Classification

## Overview

Stage A implements a multimodal 3D Attention U-Net architecture that jointly performs hippocampus segmentation and Clinical Dementia Rating (CDR) classification. This model fuses 3D MRI volumes with clinical tabular data (e.g., age, MMSE, brain volume metrics) to enable dual-task learning. It serves as a strong baseline incorporating attention gates and multimodal fusion, not a vanilla U-Net.

## Architecture

### Model: Attention U-Net 3D with Multimodal Classification Head

    The architecture extends U-Net with attention gates in the decoder and a fused multimodal classification branch:

    Encoder: 4-level 3D convolutional blocks with max pooling (base_filters = 32 → 256)
    Bottleneck: Deep feature extractor with base_filters * 16 = 512 channels
    Attention Gates: Applied at each skip connection to suppress irrelevant features
    Decoder: 3D transposed convolutions with attention-weighted skip connections
    Multimodal Fusion:
    Global average pooling of bottleneck features
    Concatenated with encoded tabular features (tab_input)
    Dual Outputs:
    Segmentation: Sigmoid output for binary hippocampus mask
    Classification: Softmax over 4 CDR classes

### Key Components

```python
def build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=7,  # e.g., ["Age", "MMSE", "eTIV", "nWBV", "ASF", "M/F", "Age_cat"]
    base_filters=32,
    num_classes=4
):
    img_in = Input(shape=input_shape, name="img_input")
    
    # Encoder
    c1 = conv_block(img_in, base_filters)
    p1 = MaxPooling3D((2,2,2))(c1)
    c2 = conv_block(p1, base_filters*2)
    p2 = MaxPooling3D((2,2,2))(c2)
    c3 = conv_block(p2, base_filters*4)
    p3 = MaxPooling3D((2,2,2))(c3)
    c4 = conv_block(p3, base_filters*8)
    p4 = MaxPooling3D((2,2,2))(c4)
    
    # Bottleneck
    bn = conv_block(p4, base_filters*16)
    bn_vec = GlobalAveragePooling3D()(bn)
    
    # Tabular input fusion
    tab_in = Input(shape=(tabular_dim,), name="tab_input")
    t = Dense(64, activation='relu')(tab_in)
    fused = Concatenate()([bn_vec, t])
    
    # Classification head
    c = Dense(128, activation='relu')(fused)
    c = Dropout(0.2)(c)
    c = Dense(64, activation='relu')(c)
    class_out = Dense(num_classes, activation='softmax', name="cdr_class")(c)
    
    # Decoder with attention gates
    g4 = Conv3D(base_filters*8, 1, padding='same')(bn)
    a4 = attention_gate(c4, g4, base_filters*8)  # ← Attention mechanism
    u4 = Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(bn)
    u4 = Concatenate()([u4, a4])
    c5 = conv_block(u4, base_filters*8)
    
    # ... (similar for levels 3, 2, 1) ...
    
    seg_out = Conv3D(1, 1, activation='sigmoid', name="seg")(c8)
    
    return Model(inputs=[img_in, tab_in], outputs=[seg_out, class_out])
```

## Data Processing

    Ensure you have the preprocessed dataset
    - Final_tvt_Dataset.zip (Train/val/test splits (images + labels))
    - strat_subset.csv (clinical metadata)

### Input Requirements

1. **MRI Volumes**: 
    Cropped and resampled to (128, 128, 128) voxels
    Normalized using z-score: (img - mean) / std
    Format: .nii files in images/ directories

2. **Segmentation Labels**: 
    Binary masks in labels/ (0 = background, >0.5 → 1 = hippocampus)
    Same spatial shape as input MRI
3. **Tabular Clinical Features (7 dimensions)**: 
    feature_cols = ["Age", "MMSE", "eTIV", "nWBV", "ASF", "M/F", "Age_cat"]
    M/F and Hand are label-encoded
    CDR scores mapped to 4 fixed classes:
    ```
    cdr_map = {0.0: 'Healthy ', 2.0: 'Very Mild ', 1.0: 'Mild', 0.5: 'Moderate'}
    ```

    Dataset Splits (from execution logs)
```
    Split	Subjects	CDR Classes Present
    Train	81	Healthy, Moderate
    Val	10	Healthy, Moderate
    Test	11	Healthy, Moderate
```

### Training Configuration
    Hyperparameters (Actual from Code)
    Input Shape: (128, 128, 128, 1)
    Batch Size: 4
    Epochs: 20
    Optimizer: AdamW (not Adam) with:
    learning_rate = 1e-3 or 2e-3
    weight_decay = 1e-4 or 1e-5
    Loss Functions:
    Segmentation: Tversky loss (α=0.5, β=0.5) → equivalent to Dice loss
    Classification: Categorical crossentropy
    Loss Weights: {'seg': 1.0, 'cdr_class': 1.0}
    Metrics:
    Segmentation: dice_coef, jaccard_coef
    Classification: accuracy
    Training Example (from log)
```python
optimizer = AdamW(learning_rate=2e-3, weight_decay=1e-5)
model.compile(
    optimizer=optimizer,
    loss={
        'seg': make_tversky_loss(0.5, 0.5),
        'cdr_class': 'categorical_crossentropy'
    },
    loss_weights={'seg': 1.0, 'cdr_class': 1.0},
    metrics={'seg': [dice_coef, jaccard_coef], 'cdr_class': ['accuracy']}
)

history = model.fit(
    train_ds.repeat(),
    epochs=20,
    steps_per_epoch=ceil(81/4)=21,
    validation_data=val_ds,
    validation_steps=ceil(11/4)=3
)
```

## Evaluation Results (from Validation Run)
    With AdamW(lr=2e-3, wd=1e-5)Segmentation:
    Mean Dice: 0.7679
    Mean Jaccard (IoU): 0.6380
    Classification:
    Accuracy: 0.8000 (8/10 correct)
    All predictions were "Healthy", correctly classifying 8 Healthy subjects but misclassifying 2 Moderate as Healthy

##### Key Features
###### What’s Actually Implemented
    3D Attention U-Net (not standard U-Net)
    Multimodal fusion of MRI + clinical tabular data
    Dual-task learning: segmentation + 4-class CDR classification
    Tversky/Dice loss for segmentation
    Fixed CDR class ordering for consistent evaluation
    Reproducibility: fixed random seeds (SEED=42)
    Comprehensive evaluation: overlays, confusion matrix, per-subject CSV

### Reproducibility

    Dependencies (from imports)

```python
import tensorflow as tf          # ≥2.10 (for AdamW)
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
```
    Random Seed
```python
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

    Files Structure
```
Stage A/
     ├── README.md               # This documentation
     └── STAGE_A_final.ipynb    # Complete implementation 
     └── val_results_Adam_lr_3e-4.zip    # AdamW(lr=3e-4) validation results
     └── val_results_AdamW_lr_1e-3_wd_1e-4.zip    # AdamW(lr=1e-3, wd=1e-4) validation results
     └── val_results_AdamW_lr_2e-3_wd_1e-.zip    # AdamW(lr=2e-3, wd=1e-5) validation results
```
    
### Conclusion
    Stage A establishes a strong multimodal baseline using attention mechanisms and clinical data fusion, far beyond a simple U-Net. It achieves high segmentation performance (Dice > 0.76) but reveals challenges in CDR classification generalization, likely due to limited class diversity in small validation sets. This sets the stage for Stage B to explore improved classification strategies or data balancing.
---

**Note**: Stage A serves as the foundation for the entire research pipeline. Ensure thorough validation before proceeding to advanced stages.