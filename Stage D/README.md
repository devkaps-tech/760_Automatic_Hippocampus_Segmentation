# Stage D: Joint Learning Framework for Simultaneous Segmentation and Classification

## Overview

    Stage D represents the culmination of the research pipeline, implementing a sophisticated joint learning framework that simultaneously performs hippocampus segmentation and CDR (Clinical Dementia Rating) classification. This stage combines all previous innovations: attention mechanisms (Stage B), multimodal fusion (Stage C), and introduces advanced joint optimization strategies for optimal performance on both tasks.

## Architecture

### Joint Multi-Task Learning Framework

    The final model architecture integrates:

1. **Shared Encoder**: Multimodal feature extraction with attention mechanisms
2. **Task-Specific Decoders**: Specialized branches for segmentation and classification
3. **Attention U-Net Backbone**: Enhanced with attention gates for better feature localization
4. **Tversky Loss Optimization**: Specialized loss function for handling class imbalance

### Model Variants

    Stage D includes three implementation variants:

#### STAGE_D_V2_final (Recommended)
- **Dataset**: Final_tvt_Dataset_v2
- **Training Samples**: 81 subjects
- **Validation Samples**: 10 subjects
- **Test Samples**: 11 subjects
- **Best Segmentation Performance**: 
  - Mean Dice: 0.7943
  - Mean Jaccard: 0.6899
- **Classification Accuracy**: 90.91%
- **Architecture**: Attention U-Net with multimodal fusion

#### STAGE_D_DV_final (Alternative Dataset)
- **Dataset**: Final_tvt_Dataset (original split)
- **Training Samples**: 81 subjects
- **Validation Samples**: 11 subjects
- **Test Samples**: 11 subjects
- **Best Segmentation Performance**:
  - Mean Dice: 0.8202
  - Mean Jaccard: 0.7019
- **Classification Accuracy**: 90.91%

#### STAGE_D_DV1_final (Version 1 Dataset)
- **Dataset**: Final_tvt_Dataset_v1
- **Training Samples**: 81 subjects
- **Validation Samples**: 10 subjects
- **Test Samples**: 11 subjects
- **Best Segmentation Performance**:
  - Mean Dice: 0.8678
  - Mean Jaccard: 0.7689
- **Classification Accuracy**: 81.82%

## Advanced Joint Architecture

### Attention U-Net with Classification Branch
```python
def build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=7,  # Clinical features
    base_filters=32,
    class_hidden=(128,64),
    dropout_rate=0.2,
    num_classes=4  # Healthy, Very Mild, Mild, Moderate
):
    """
    Attention U-Net with classification branch
    """
    # Image input
    img_in = layers.Input(shape=input_shape, name="img_input")
    
    # Encoder with attention
    c1 = conv_block(img_in, base_filters, dropout_rate)
    p1 = layers.MaxPooling3D((2,2,2))(c1)
    
    c2 = conv_block(p1, base_filters*2, dropout_rate)
    p2 = layers.MaxPooling3D((2,2,2))(c2)
    
    c3 = conv_block(p2, base_filters*4, dropout_rate)
    p3 = layers.MaxPooling3D((2,2,2))(c3)
    
    c4 = conv_block(p3, base_filters*8, dropout_rate)
    p4 = layers.MaxPooling3D((2,2,2))(c4)
    
    # Bottleneck
    bn = conv_block(p4, base_filters*16, dropout_rate)
    bn_vec = layers.GlobalAveragePooling3D(name="bn_gap")(bn)
    
    # Tabular input fusion
    if tabular_dim > 0:
        tab_in = layers.Input(shape=(tabular_dim,), name="tab_input")
        t = layers.Dense(64, activation='relu')(tab_in)
        fused = layers.Concatenate()([bn_vec, t])
        inputs = [img_in, tab_in]
    else:
        fused = bn_vec
        inputs = [img_in]
    
    # Classification branch
    c = fused
    for i, units in enumerate(class_hidden):
        c = layers.Dense(units, activation='relu')(c)
        c = layers.Dropout(dropout_rate)(c)
    class_out = layers.Dense(num_classes, activation='softmax', name="cdr_class")(c)
    
    # Decoder with attention gates
    g4 = layers.Conv3D(base_filters*8, 1, padding='same')(bn)
    a4 = attention_gate(c4, g4, base_filters*8)
    u4 = layers.Conv3DTranspose(base_filters*8, 2, strides=2, padding='same')(bn)
    u4 = layers.Concatenate()([u4, a4])
    c5 = conv_block(u4, base_filters*8, dropout_rate)
    
    g3 = layers.Conv3D(base_filters*4, 1, padding='same')(c5)
    a3 = attention_gate(c3, g3, base_filters*4)
    u3 = layers.Conv3DTranspose(base_filters*4, 2, strides=2, padding='same')(c5)
    u3 = layers.Concatenate()([u3, a3])
    c6 = conv_block(u3, base_filters*4, dropout_rate)
    
    g2 = layers.Conv3D(base_filters*2, 1, padding='same')(c6)
    a2 = attention_gate(c2, g2, base_filters*2)
    u2 = layers.Conv3DTranspose(base_filters*2, 2, strides=2, padding='same')(c6)
    u2 = layers.Concatenate()([u2, a2])
    c7 = conv_block(u2, base_filters*2, dropout_rate)
    
    g1 = layers.Conv3D(base_filters, 1, padding='same')(c7)
    a1 = attention_gate(c1, g1, base_filters)
    u1 = layers.Conv3DTranspose(base_filters, 2, strides=2, padding='same')(c7)
    u1 = layers.Concatenate()([u1, a1])
    c8 = conv_block(u1, base_filters, dropout_rate)
    
    # Segmentation output
    seg_out = layers.Conv3D(1, 1, activation='sigmoid', name="seg")(c8)
    
    model = models.Model(inputs=inputs, outputs=[seg_out, class_out])
    return model
```

### Attention Gate Mechanism
```python
def attention_gate(x, g, inter_channels):
    """
    Attention gate for feature selection
    x: skip connection features
    g: gating signal from coarser scale
    """
    theta_x = layers.Conv3D(inter_channels, 2, strides=2, padding='same')(x)
    phi_g = layers.Conv3D(inter_channels, 1, padding='same')(g)
    
    add = layers.Add()([theta_x, phi_g])
    act = layers.ReLU()(add)
    
    psi = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(act)
    upsampled_psi = layers.UpSampling3D(size=(2,2,2))(psi)
    
    attn_out = layers.Multiply()([x, upsampled_psi])
    return attn_out
```

## Training Configuration

### Loss Functions
```python
def make_tversky_loss(alpha=0.4, beta=0.6, smooth=1e-6):
    """
    Tversky loss for handling class imbalance
    alpha: weight for false positives
    beta: weight for false negatives
    """
    def loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        tp = K.sum(y_true_f * y_pred_f)
        fp = K.sum((1 - y_true_f) * y_pred_f)
        fn = K.sum(y_true_f * (1 - y_pred_f))
        
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1 - tversky
    return loss

# Combined loss for joint training
model.compile(
    optimizer=AdamW(learning_rate=2e-3, weight_decay=1e-5),
    loss={
        'seg': make_tversky_loss(alpha=0.4, beta=0.6),
        'cdr_class': 'categorical_crossentropy'
    },
    loss_weights={'seg': 0.9, 'cdr_class': 1.1},
    metrics={
        'seg': [dice_coef, jaccard_coef],
        'cdr_class': ['accuracy']
    }
)
```

### Data Preprocessing
```python
# Clinical features used in all variants
feature_cols = ["Age", "MMSE", "eTIV", "nWBV", "ASF", "M/F", "Age_cat"]

# CDR classification mapping
cdr_map = {
    0.0: 'Healthy',      # CDR_Class: 0
    2.0: 'Very Mild',    # CDR_Class: 1
    1.0: 'Mild',         # CDR_Class: 2
    0.5: 'Moderate'      # CDR_Class: 3
}

# Image normalization
def normalize_image(img, eps=1e-8):
    return (img - np.mean(img)) / (np.std(img) + eps)
```

### Training Parameters
```python
# Training configuration
BATCH_SIZE = 4
EPOCHS = 50
dropout_rate = 0.2
alpha, beta = 0.4, 0.6  # Tversky loss parameters

# Learning rate schedule
optimizer = AdamW(learning_rate=2e-3, weight_decay=1e-5)

# Steps calculation
steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
validation_steps = math.ceil(n_val / BATCH_SIZE)
```

## Performance Comparison

### Segmentation Results

| Variant | Dataset Version | Mean Dice | Mean Jaccard | Test Accuracy |
|---------|----------------|-----------|--------------|---------------|
| **STAGE_D_DV1** | v1 | **0.8678** | **0.7689** | 81.82% |
| **STAGE_D_DV** | original | 0.8202 | 0.7019 | **90.91%** |
| **STAGE_D_V2** | v2 | 0.7943 | 0.6899 | **90.91%** |

### Classification Results

All variants achieved high classification accuracy for Healthy subjects:
- **Precision**: 0.9091 (V2, DV) / 0.8182 (DV1)
- **Recall**: 1.0000 (all variants)
- **F1-Score**: 0.9524 (V2, DV) / 0.9000 (DV1)

**Note**: Limited representation of Very Mild, Mild, and Moderate classes in test set affected per-class metrics.

## Training History Analysis

### Convergence Patterns (STAGE_D_V2)

**Segmentation Performance**:
- Initial Dice: 0.0137 → Final Dice: 0.8880
- Initial Jaccard: 0.0069 → Final Jaccard: 0.7987
- Validation stabilized after epoch 30

**Classification Performance**:
- Training accuracy improved from 56.53% to 68.39%
- Validation accuracy: 100% (consistent from epoch 3)
- Classification loss reduced significantly over training

### Loss Progression
```python
# Final epoch metrics (Epoch 50, V2)
Segmentation:
  - seg_loss: 0.1125
  - val_seg_loss: 0.0976
  - dice_coef: 0.8804
  - val_dice_coef: 0.8880

Classification:
  - cdr_class_loss: 0.7817
  - val_cdr_class_loss: 0.4265
  - cdr_class_accuracy: 0.7631
  - val_cdr_class_accuracy: 1.0000
```

## Implementation Details

### Data Generator
```python
def multimodal_generator(img_paths, lbl_paths, ids, df, feature_cols,
                        target_col="CDR_Class", num_classes=4):
    """
    Generator for joint segmentation and classification
    """
    df_indexed = df.set_index("ID_num")
    
    for img_p, lbl_p, sid in zip(img_paths, lbl_paths, ids):
        # Load and preprocess image
        img = load_nifti(img_p).astype(np.float32)
        lbl = load_nifti(lbl_p).astype(np.float32)
        
        img = normalize_image(img)
        lbl = (lbl > 0.5).astype(np.float32)
        
        # Add channel dimension
        img = img[..., np.newaxis]
        lbl = lbl[..., np.newaxis]
        
        # Get clinical features
        feats = df_indexed.loc[sid, feature_cols].to_numpy().astype(np.float32)
        
        # Get classification target
        targ_class = int(df_indexed.loc[sid, target_col])
        targ_onehot = tf.keras.utils.to_categorical(targ_class, num_classes)
        
        yield ({"img_input": img, "tab_input": feats},
               {"seg": lbl, "cdr_class": targ_onehot})
```

### Evaluation Metrics
```python
def test_and_save_multimodal_classification(
    model, test_pairs, df_features, feature_cols,
    target_col="CDR_Class", out_dir="test_results",
    threshold=0.5, num_classes=4,
    class_names=('Healthy','Very Mild','Mild','Moderate')
):
    """
    Comprehensive evaluation of joint model
    """
    dice_scores, jaccard_scores = [], []
    y_true_cls, y_pred_cls = [], []
    
    for img_path, lbl_path in test_pairs:
        # Predict segmentation and classification
        seg_pred, class_pred = model.predict(...)
        
        # Calculate segmentation metrics
        dice = dice_coef_np(lbl, pred_bin)
        jaccard = jaccard_coef_np(lbl, pred_bin)
        
        # Store results
        dice_scores.append(dice)
        jaccard_scores.append(jaccard)
        y_true_cls.append(true_class)
        y_pred_cls.append(pred_class)
    
    # Generate comprehensive report
    mean_dice = np.mean(dice_scores)
    mean_jaccard = np.mean(jaccard_scores)
    accuracy = accuracy_score(y_true_cls, y_pred_cls)
    
    # Create confusion matrix and classification report
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    cls_report = classification_report(y_true_cls, y_pred_cls, 
                                       target_names=class_names)
    
    return results
```

## Usage Instructions

### Step 1: Environment Setup
```bash
# Install required packages
pip install tensorflow==2.x
pip install nibabel pandas scikit-learn matplotlib seaborn
```

### Step 2: Data Preparation
```python
# Load and preprocess CSV data
df = pd.read_csv("strat_subset.csv")
df["ID_num"] = df["ID"].str.split("_").str[1]

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

def preprocess_csv(df, le_dict=None, fit=True):
    cat_cols = ["M/F", "Hand"]
    # ... encoding logic ...
    
    # Map CDR to classification labels
    cdr_map = {0.0: 'Healthy', 2.0: 'Very Mild', 
               1.0: 'Mild', 0.5: 'Moderate'}
    df['CDR_Label'] = df['CDR'].map(cdr_map)
    
    return df, le_dict
```

### Step 3: Model Training
```python
# Build model
model = build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=len(feature_cols),
    base_filters=32,
    dropout_rate=0.2,
    num_classes=4
)

# Compile
model.compile(
    optimizer=AdamW(learning_rate=2e-3, weight_decay=1e-5),
    loss={
        'seg': make_tversky_loss(alpha=0.4, beta=0.6),
        'cdr_class': 'categorical_crossentropy'
    },
    loss_weights={'seg': 0.9, 'cdr_class': 1.1},
    metrics={
        'seg': [dice_coef, jaccard_coef],
        'cdr_class': ['accuracy']
    }
)

# Train
history = model.fit(
    train_ds.repeat(),
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    batch_size=BATCH_SIZE,
    verbose=1
)
```

### Step 4: Evaluation
```python
# Test model
test_results = test_and_save_multimodal_classification(
    model=model,
    test_pairs=test_pairs,
    df_features=test_df,
    feature_cols=feature_cols,
    target_col="CDR_Class",
    out_dir="test_results",
    threshold=0.5,
    num_classes=4,
    class_names=['Healthy','Very Mild','Mild','Moderate']
)

print(f"Mean Dice: {test_results['mean_dice']:.4f}")
print(f"Mean Jaccard: {test_results['mean_jaccard']:.4f}")
print(f"Classification Accuracy: {test_results['accuracy']:.4f}")
```

## Files Structure
```
Stage D/
├── README.md                       # This documentation
├── STAGE_D_V2_final.ipynb           # V2 implementation (recommended)
├── STAGE_D_DV_final.ipynb           # Original dataset variant
├── STAGE_D_DV1_final.ipynb          # V1 dataset variant
└── strat_subset.csv                   # loading data
└── test_results_E50.zip                   # Original results
└── test_results_v1_E50.zip                  # V2 results
└── test_results_v2_E50.zip                  # V1 results
```

## Key Findings

### Strengths
1. **High Segmentation Accuracy**: Dice coefficients > 0.79 across all variants
2. **Excellent Healthy Classification**: 90-91% accuracy for primary class
3. **Effective Multimodal Fusion**: Clinical features improve performance
4. **Attention Mechanisms**: Enhance feature localization

### Limitations
1. **Class Imbalance**: Limited samples for Very Mild, Mild, Moderate classes
2. **Dataset Size**: 81 training samples limits generalization
3. **Single Site Data**: OASIS-1 only, needs multi-site validation

### Recommendations
1. **Increase Sample Size**: Incorporate additional datasets (ADNI, AIBL)
2. **Address Class Imbalance**: Use data augmentation and balanced sampling
3. **Hyperparameter Optimization**: Fine-tune loss weights and architecture
4. **Longitudinal Analysis**: Extend to temporal progression prediction

## Clinical Implications

### Diagnostic Support
- Automated hippocampus segmentation reduces manual annotation time
- CDR classification provides cognitive assessment support
- Combined analysis links structural changes to cognitive decline

### Research Applications
- Biomarker discovery for early Alzheimer's detection
- Treatment response monitoring
- Clinical trial patient stratification

## Conclusion

    Stage D demonstrates the effectiveness of joint learning for simultaneous hippocampus segmentation and cognitive assessment. The attention-enhanced multimodal architecture achieves strong performance on both tasks, with STAGE_D_DV1 showing the best segmentation results (Dice: 0.8678) and STAGE_D_V2/DV achieving the highest classification accuracy (90.91%).

    The framework successfully integrates MRI imaging with clinical data, providing a comprehensive approach to automated Alzheimer's disease assessment. Future work should focus on expanding the dataset, addressing class imbalance, and validating across multiple imaging sites.

---

**Note**: All three variants are fully implemented and ready for execution. Choose the variant based on your specific dataset and research objectives.