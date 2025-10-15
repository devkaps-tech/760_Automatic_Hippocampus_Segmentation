# Stage C: Multimodal Fusion Model for Enhanced Hippocampus Analysis

## Overview

    Stage C introduces a multi-task learning framework that simultaneously performs hippocampus segmentation and CDR (Clinical Dementia Rating) classification by fusing 3D MRI volumes with clinical tabular data. The model uses an Attention U-Net 3D backbone enhanced with a tabular feature fusion branch. The clinical data includes demographics, cognitive scores, and brain measurements from the strat_subset.csv file.

## Architecture

### Multimodal Fusion Framework

    The implemented model integrates two data modalities:

    3D MRI Branch: Attention U-Net encoder-decoder for segmentation
    Clinical Data Branch: Dense network processing 7 selected tabular features
    Fusion Module: Late fusion at the bottleneck via concatenation
    Multi-task Outputs:
    Segmentation mask (sigmoid output)
    CDR class prediction (4-class softmax: Healthy, Very Mild, Mild, Moderate)

    Actual Model Architecture (from document)
```python
def build_att_unet3d_with_class(
    input_shape=(128,128,128,1), 
    tabular_dim=7,  # Exactly 7 features used
    base_filters=32, 
    class_hidden=(128,64), 
    dropout_rate=0.2, 
    num_classes=4
):
    img_in = layers.Input(shape=input_shape, name="img_input")
    
    # Encoder (U-Net down path with attention gates in decoder)
    c1 = conv_block(img_in, base_filters, dropout_rate)
    p1 = layers.MaxPooling3D((2,2,2))(c1)
    c2 = conv_block(p1, base_filters*2, dropout_rate)
    p2 = layers.MaxPooling3D((2,2,2))(c2)
    c3 = conv_block(p2, base_filters*4, dropout_rate)
    p3 = layers.MaxPooling3D((2,2,2))(c3)
    c4 = conv_block(p3, base_filters*8, dropout_rate)
    p4 = layers.MaxPooling3D((2,2,2))(c4)
    bn = conv_block(p4, base_filters*16, dropout_rate)  # Bottleneck
    
    # Global feature vector from bottleneck
    bn_vec = layers.GlobalAveragePooling3D(name="bn_gap")(bn)
    
    # Clinical data branch (7 features)
    tab_in = layers.Input(shape=(tabular_dim,), name="tab_input")
    t = layers.Dense(64, activation='relu', name="tab_dense")(tab_in)
    
    # Late fusion: concatenate bottleneck features with processed tabular features
    fused = layers.Concatenate(name="fuse_bn_tab")([bn_vec, t])
    
    # Classification head
    c = fused
    for i, u in enumerate(class_hidden):
        c = layers.Dense(u, activation='relu', name=f"class_dense_{i+1}")(c)
        c = layers.Dropout(dropout_rate)(c)
    class_out = layers.Dense(num_classes, activation='softmax', name="cdr_class")(c)
    
    # Decoder with attention gates (up path)
    g4 = layers.Conv3D(base_filters*8, 1, padding='same')(bn)
    a4 = attention_gate(c4, g4, base_filters*8)
    u4 = layers.Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(bn)
    u4 = layers.Concatenate()([u4, a4])
    c5 = conv_block(u4, base_filters*8, dropout_rate)
    # ... (similar attention blocks for g3/a3, g2/a2, g1/a1)
    
    c8 = conv_block(u1, base_filters, dropout_rate)
    seg_out = layers.Conv3D(1, 1, activation='sigmoid', name="seg")(c8)
    
    model = models.Model(inputs=[img_in, tab_in], outputs=[seg_out, class_out])
    return model
```

## Clinical Data Processing

### Feature Selection (from document)
    The following 7 features are used from the CSV:

```python
feature_cols = ["Age", "MMSE", "eTIV", "nWBV", "ASF", "M/F", "Age_cat"]
```

### Data Preprocessing Pipeline
    1.CSV Processing:
    Extract ID_num from ID column (e.g., "OASIS_0009_MR1" → "0009")
    Filter CSV to match available NIfTI label files in train/val/test splits
    Remove duplicate subjects (keep first MR session)
    2.Categorical Encoding:
    M/F and Hand: Label-encoded
    CDR: Mapped to 4 fixed classes with enforced order:
```python
cdr_map = {0.0: 'Healthy ', 2.0: 'Very Mild ', 1.0: 'Mild', 0.5: 'Moderate'}
```
    3.Dataset Splits (confirmed from output):
    Train: 81 subjects
    Validation: 10 subjects
    Test: 11 subjects

## Training Configuration
    Loss Functions (from document)

```python
# Segmentation: Tversky Loss (configurable alpha/beta)
def make_tversky_loss(alpha=0.7, beta=0.3, smooth=1e-6):
    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum((1 - y_true_f) * y_pred_f)
        fn = K.sum(y_true_f * (1 - y_pred_f))
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky
    return loss

# Classification: Categorical Crossentropy
```

### Training Setup (from document)
    Optimizer: AdamW (learning_rate=2e-3, weight_decay=1e-5)
    Batch Size: 4
    Loss Weights: {'seg': 0.9, 'cdr_class': 1.1}
    Dropout Rate: 0.2
    Epochs: 20
#### Metrics:
    Segmentation: Dice Coefficient, Jaccard Coefficient
    Classification: Accuracy

### Tversky Parameter Experiments
    Three configurations were tested:

    alpha=0.5, beta=0.5 (balanced)
    alpha=0.3, beta=0.7 (penalize false negatives more)
    alpha=0.4, beta=0.6 (optimal balance)

### Data Pipeline
    Multimodal Data Generator

```python
def multimodal_generator(img_paths, lbl_paths, ids, df, feature_cols, target_col="CDR_Class", num_classes=4):
    df_indexed = df.set_index("ID_num")
    for img_p, lbl_p, sid in zip(img_paths, lbl_paths, ids):
        # Load and normalize MRI
        img = load_nifti(img_p).astype(np.float32)
        img = normalize_image(img)  # (img - mean) / std
        img = img[..., np.newaxis]
        
        # Load and binarize label
        lbl = load_nifti(lbl_p).astype(np.float32)
        lbl = (lbl > 0.5).astype(np.float32)[..., np.newaxis]
        
        # Extract tabular features
        feats = df_indexed.loc[sid, feature_cols].to_numpy().astype(np.float32)
        
        # One-hot encode CDR class
        targ_class = int(df_indexed.loc[sid, target_col])
        targ_onehot = tf.keras.utils.to_categorical(targ_class, num_classes=num_classes).astype(np.float32)
        
        yield ({"img_input": img, "tab_input": feats}, 
               {"seg": lbl, "cdr_class": targ_onehot})
```



### Evaluation Results
    Validation Results (from document)
    For the best configuration (alpha=0.4, beta=0.6):

    Segmentation:
    Mean Dice: 0.8655
    Mean Jaccard: 0.7632
    Classification:
    Accuracy: 0.8000 (8/10 correct)
    Confusion Matrix

```
[[8, 0, 0, 0],  # All Healthy correctly predicted
 [0, 0, 0, 0],
 [0, 0, 0, 0],
 [2, 0, 0, 0]]  # Both Moderate cases misclassified as Healthy
```
    Class Distribution: Only Healthy (class 0) and Moderate (class 3) present in validation set


## Key Observations
    1. Segmentation Performance: High Dice scores (>0.85) indicate excellent hippocampus delineation
    2. Classification Challenge: Model struggles with Moderate dementia cases (predicts all as Healthy)
    3. Data Imbalance: No "Very Mild" or "Mild" cases in validation set limits classification evaluation

### Implementation Details
    File Structure (from document)
```
Stage C/
├── README.md                    # This documentation
├── STAGE_C_final.ipynb         # Complete multimodal implementation
├── val_results_AdamW_lw_0.9-1.1_D0.2_A0.3_B0.7.zip        # AdamW, lw=0.9-1.1, D=0.2, A=0.3, B=0.7 results
├── val_results_AdamW_lw_0.9-1.1_D0.2_A0.4_B0.6.zip      # AdamW, lw=0.9-1.1, D=0.2, A=0.4, B=0.6 results
└── val_results_AdamW_lw_0.9-1.1_D0.2.zip        # AdamW, lw=0.9-1.1, D=0.2, default A=0.5, B=0.5 results
```

    Critical Code Elements
    Input Shape: Fixed to (128, 128, 128, 1) for all MRI volumes
    Attention Gates: Implemented between encoder and decoder skip connections
    Late Fusion: Tabular features fused only at bottleneck (not early/intermediate)
    No Regression Task: Only segmentation + classification (no MMSE regression as originally described)
    Fixed Class Order: CDR classes enforced as ['Healthy ', 'Very Mild ', 'Mild', 'Moderate']

### Usage Instructions
    To Reproduce Results

##### 1.Prepare Data:

```python
unzip_to_folder("/content/Final_tvt_Dataset.zip")
df = pd.read_csv("strat_subset.csv")
```

##### 2.Process Splits (as shown in document for train/val/test)
##### 3.Build and Train Model

```python
model = build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=7,
    dropout_rate=0.2
)
model.compile(
    optimizer=AdamW(learning_rate=2e-3, weight_decay=1e-5),
    loss={'seg': make_tversky_loss(0.4, 0.6), 'cdr_class': 'categorical_crossentropy'},
    loss_weights={'seg': 0.9, 'cdr_class': 1.1},
    metrics={'seg': [dice_coef, jaccard_coef], 'cdr_class': ['accuracy']}
)
```
##### 4.Evaluate
```python
val_results = test_and_save_multimodal_classification(
    model, val_pairs, val_df, feature_cols,
    out_dir="val_results", num_classes=4
)
```

## Conclusion
    The implemented Stage C demonstrates:
    - Successful fusion of 3D MRI and clinical tabular data
    - High-quality hippocampus segmentation (Dice > 0.86)
    -Classification limitations due to data imbalance in validation set
    -Flexible Tversky loss for segmentation optimization

--- 
**Note**: The actual implementation focuses on segmentation + classification rather than the originally described regression tasks, with clinical features fused only at the bottleneck (late fusion) rather than multiple fusion strategies.

