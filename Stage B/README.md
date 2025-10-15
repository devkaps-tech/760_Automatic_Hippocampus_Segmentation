# Stage B: Loss-Weighted Multimodal 3D Attention U-Net for Hippocampus Segmentation and CDR Classification

## Overview

    Stage B introduces **loss weighting strategies** to address task imbalance between segmentation and classification, as well as **class imbalance** in the CDR classification task. The architecture remains identical to Stage D (3D Attention U-Net with multimodal fusion), but explores different loss weighting configurations and weighted classification loss functions.

**Key Innovation**: Rather than architectural changes, Stage B focuses on training strategy optimization through:
- Multiple loss weight configurations
- Weighted categorical cross-entropy for class imbalance
- Systematic comparison of training approaches

## Architecture

### Model: 3D Attention U-Net with CDR Classification

    The model architecture is identical to Stage D, consisting of:

1. **Attention U-Net Encoder**: 3D convolution blocks with attention gates
2. **Bottleneck**: Global average pooling for feature extraction
3. **Multimodal Fusion**: Integration of imaging and clinical features
4. **Dual Task Decoders**:
   - Segmentation branch with attention-guided skip connections
   - Classification branch for CDR prediction
```python
def build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=7,  # Clinical features
    base_filters=32,
    class_hidden=(128,64),
    dropout_rate=0.2,  # Note: Also tested with dropout=0
    num_classes=4  # Healthy, Very Mild, Mild, Moderate
):
    """
    Attention U-Net with classification branch
    Same architecture as Stage D
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
    
    # Multimodal fusion
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
    
    # Decoder with attention gates (same as Stage D)
    # ... [decoder implementation] ...
    
    # Segmentation output
    seg_out = layers.Conv3D(1, 1, activation='sigmoid', name="seg")(c8)
    
    model = models.Model(inputs=inputs, outputs=[seg_out, class_out])
    return model
```

### Attention Gate Mechanism
```python
def attention_gate(x, g, inter_channels):
    """
    Attention gate for skip connections
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

## Loss Weighting Experiments

### Experiment 1: Loss Weights {seg: 0.8, cdr_class: 1.2}

**Configuration**:
```python
dropout_rate = 0
alpha, beta = 0.5, 0.5  # Tversky loss balance
BATCH_SIZE = 4
optimizer = AdamW(learning_rate=2e-3, weight_decay=1e-5)

model.compile(
    optimizer=optimizer,
    loss={
        'seg': make_tversky_loss(alpha=0.5, beta=0.5),
        'cdr_class': 'categorical_crossentropy'
    },
    loss_weights={'seg': 0.8, 'cdr_class': 1.2},
    metrics={
        'seg': [dice_coef, jaccard_coef],
        'cdr_class': ['accuracy']
    }
)
```

**Training Results** (20 epochs):
- **Validation Dice**: 0.6615
- **Validation Jaccard**: 0.4948
- **Validation Classification Accuracy**: 0% (highly unstable)

**Final Test Results**:
```
Mean Dice: 0.7147
Mean Jaccard: 0.5596
Classification Accuracy: 0.2000
```

**Confusion Matrix**:
- Predicted all samples as Moderate (class 3)
- Poor classification performance due to weight imbalance

### Experiment 2: Loss Weights {seg: 0.9, cdr_class: 1.1}

**Configuration**:
```python
dropout_rate = 0
alpha, beta = 0.5, 0.5
loss_weights={'seg': 0.9, 'cdr_class': 1.1}
```

**Training Results** (20 epochs):
- **Validation Dice**: 0.5539
- **Validation Jaccard**: 0.4059
- **Validation Classification Accuracy**: 100%

**Final Test Results**:
```
Mean Dice: 0.6136
Mean Jaccard: 0.4558
Classification Accuracy: 0.8000 (8/10 correct)
```

**Confusion Matrix**:
```
[[8, 0, 0, 0],    # Healthy: 8/8 correct
 [0, 0, 0, 0],    # Very Mild: none in test set
 [0, 0, 0, 0],    # Mild: none in test set
 [2, 0, 0, 0]]    # Moderate: 0/2 correct (misclassified as Healthy)
```

**Performance Metrics**:
- **Healthy Class**: Precision 0.8000, Recall 1.0000, F1-Score 0.8889
- **Moderate Class**: Precision 0.0000, Recall 0.0000, F1-Score 0.0000

### Experiment 3: Weighted Categorical Cross-Entropy

**Class Distribution Analysis**:
```python
# Training set class distribution
TRAIN class counts [0..3]: [55, 1, 5, 20]
# Healthy: 55, Very Mild: 1, Mild: 5, Moderate: 20

# Computed inverse-frequency weights (normalized)
class_weights = [0.0573, 3.1541, 0.6308, 0.1577]
```

**Weighted Loss Implementation**:
```python
def weighted_categorical_crossentropy(class_weights, epsilon=1e-7):
    """
    Apply class-specific weights to address imbalance
    """
    w = tf.constant(class_weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        logp = tf.math.log(y_pred)
        
        # Weight per true class
        w_per_sample = tf.reduce_sum(y_true * w, axis=-1)
        ce = -tf.reduce_sum(y_true * logp, axis=-1)
        
        return tf.reduce_mean(w_per_sample * ce)
    return loss
```

**Configuration**:
```python
clf_loss = weighted_categorical_crossentropy(class_weights)

model.compile(
    optimizer=optimizer,
    loss={
        'seg': make_tversky_loss(alpha=0.5, beta=0.5),
        'cdr_class': clf_loss  # Weighted loss
    },
    loss_weights={'seg': 0.9, 'cdr_class': 1.1},
    metrics={
        'seg': [dice_coef, jaccard_coef],
        'cdr_class': ['accuracy']
    }
)
```

**Training Results** (20 epochs):
- **Validation Dice**: 0.6697
- **Validation Jaccard**: 0.5051
- **Validation Classification Accuracy**: 0% (unstable)

**Final Test Results**:
```
Mean Dice: 0.7102
Mean Jaccard: 0.5530
Classification Accuracy: 0.0000 (0/10 correct)
```

**Confusion Matrix**:
```
[[0, 0, 8, 0],    # Healthy: misclassified as Mild
 [0, 0, 0, 0],    # Very Mild: none in test set
 [0, 0, 0, 0],    # Mild: none in test set  
 [0, 0, 2, 0]]    # Moderate: misclassified as Mild
```

**Issue**: Weighted loss caused model to predict all samples as Mild (class 2), likely due to the high weight assigned to the Very Mild class (3.15) with only 1 training sample.

## Data Configuration

### Dataset Split
- **Training**: 81 subjects
- **Validation**: 10 subjects  
- **Test**: 11 subjects

### Clinical Features
```python
feature_cols = ["Age", "MMSE", "eTIV", "nWBV", "ASF", "M/F", "Age_cat"]
# 7 features total
```

### CDR Classification Mapping
```python
cdr_map = {
    0.0: 'Healthy',      # CDR_Class: 0
    2.0: 'Very Mild',    # CDR_Class: 1
    1.0: 'Mild',         # CDR_Class: 2
    0.5: 'Moderate'      # CDR_Class: 3
}
```

## Loss Functions

### Tversky Loss for Segmentation
```python
def make_tversky_loss(alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Tversky loss for handling segmentation
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
```

### Classification Loss Options

1. **Standard Categorical Cross-Entropy**:
```python
'categorical_crossentropy'
```

2. **Weighted Categorical Cross-Entropy**:
```python
weighted_categorical_crossentropy(class_weights=[0.0573, 3.1541, 0.6308, 0.1577])
```

## Training Configuration

### Common Hyperparameters
```python
SEED = 42
dropout_rate = 0  # Tested without dropout
alpha, beta = 0.5, 0.5  # Balanced Tversky loss
BATCH_SIZE = 4
EPOCHS = 20
learning_rate = 2e-3
weight_decay = 1e-5

# Steps calculation
steps_per_epoch = math.ceil(81 / 4)  # 21 steps
validation_steps = math.ceil(11 / 4)  # 3 steps
```

### Optimizer
```python
from keras.optimizers import AdamW

optimizer = AdamW(
    learning_rate=2e-3,
    weight_decay=1e-5
)
```

## Performance Comparison

### Experiment Results Summary

| Configuration | Loss Weights | Val Dice | Val Jaccard | Val Acc | Test Dice | Test Jaccard | Test Acc |
|--------------|--------------|----------|-------------|---------|-----------|--------------|----------|
| **Exp 1** | seg:0.8, cls:1.2 | 0.6615 | 0.4948 | 0% | 0.7147 | 0.5596 | 20% |
| **Exp 2** | seg:0.9, cls:1.1 | 0.5539 | 0.4059 | 100% | 0.6136 | 0.4558 | 80% |
| **Exp 3** | seg:0.9, cls:1.1 + WCE | 0.6697 | 0.5051 | 0% | 0.7102 | 0.5530 | 0% |

**Key Findings**:
1. **Best Segmentation**: Experiment 1 (Dice: 0.7147)
2. **Best Classification**: Experiment 2 (Accuracy: 80%)
3. **Most Balanced**: Experiment 2 offers best trade-off
4. **Weighted Loss Issue**: Heavy class weights caused prediction bias

### Training Dynamics

#### Experiment 2 Training Progression:
```
Epoch 1:  Dice 0.0119 → Val_Dice 0.0080
Epoch 5:  Dice 0.2072 → Val_Dice 0.0082
Epoch 10: Dice 0.7796 → Val_Dice 0.0536
Epoch 15: Dice 0.8360 → Val_Dice 0.2283
Epoch 19: Dice 0.8278 → Val_Dice 0.8064  (best)
Epoch 20: Dice 0.7876 → Val_Dice 0.5539
```

**Observation**: Validation metrics highly unstable, suggesting:
- Small validation set (10 subjects) causes high variance
- Model struggling to generalize classification
- Segmentation more stable than classification

## Evaluation Methodology

### Test Evaluation Function
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
    # Segmentation metrics
    dice_scores = []
    jaccard_scores = []
    
    # Classification metrics
    y_true_cls = []
    y_pred_cls = []
    
    for img_path, lbl_path in test_pairs:
        # Load and preprocess
        img = normalize_image(load_nifti(img_path))
        lbl = load_nifti(lbl_path)
        
        # Predict
        seg_pred, class_pred = model.predict(...)
        
        # Calculate metrics
        dice = dice_coef_np(lbl, pred_bin)
        jaccard = jaccard_coef_np(lbl, pred_bin)
        
        # Store results
        dice_scores.append(dice)
        y_true_cls.append(true_class)
        y_pred_cls.append(pred_class)
    
    # Generate comprehensive report
    return {
        'mean_dice': np.mean(dice_scores),
        'mean_jaccard': np.mean(jaccard_scores),
        'accuracy': accuracy_score(y_true_cls, y_pred_cls),
        'confusion_matrix': confusion_matrix(...),
        'classification_report': classification_report(...)
    }
```

## Key Findings and Analysis

### Loss Weighting Insights

1. **Segmentation vs Classification Balance**:
   - Higher classification weight (1.2) → Better segmentation but poor classification
   - Balanced weights (0.9/1.1) → Better overall performance
   - Loss weight ratio critically affects task balance

2. **Class Imbalance Challenges**:
   - Training set highly imbalanced (Healthy:55, Very Mild:1, Mild:5, Moderate:20)
   - Simple weighted loss can cause prediction bias
   - Small classes (Very Mild with 1 sample) problematic for weighting

3. **Training Instability**:
   - Classification accuracy highly unstable during training
   - Validation metrics fluctuate significantly
   - Small validation set (10 subjects) exacerbates instability


## Limitations and Challenges

### Major Issues Identified

1. **Class Imbalance**:
   - Very Mild class has only 1 training sample
   - Weighted loss causes severe prediction bias
   - Model tends to overfit to majority class

2. **Small Dataset**:
   - 81 training samples insufficient for robust learning
   - Validation set (10) too small for reliable metrics
   - High variance in performance estimates

3. **Task Interference**:
   - Segmentation and classification objectives may conflict
   - Loss weight tuning difficult without clear guidelines
   - No clear optimal configuration found

4. **Evaluation Challenges**:
   - Test set lacks Very Mild and Mild samples
   - Cannot assess full classification performance
   - Metrics dominated by Healthy class

### Recommendations for Improvement

1. **Data Augmentation**:
   - Synthetic oversampling for minority classes
   - Advanced augmentation techniques
   - Consider focal loss for class imbalance

2. **Training Strategy**:
   - Curriculum learning (train segmentation first)
   - Gradual loss weight adjustment
   - Longer training with early stopping

3. **Architecture Modifications**:
   - Separate encoders for each task
   - Task-specific batch normalization
   - Auxiliary losses for better gradients

4. **Evaluation Improvements**:
   - Cross-validation for robust estimates
   - Stratified sampling in data splits
   - Additional datasets for validation

## Usage Instructions

### Step 1: Environment Setup
```bash
pip install tensorflow==2.x nibabel pandas scikit-learn matplotlib seaborn
```

### Step 2: Data Preparation
```python
# Load preprocessed data
df = pd.read_csv("strat_subset.csv")
df["ID_num"] = df["ID"].str.split("_").str[1]

# Encode categorical variables and CDR labels
train_df, encoders = preprocess_csv(train_csv, fit=True)
val_df, _ = preprocess_csv(val_csv, le_dict=encoders, fit=False)
```

### Step 3: Choose Experiment Configuration
```python
# Select configuration based on priority
if prioritize_segmentation:
    loss_weights = {'seg': 0.8, 'cdr_class': 1.2}
elif prioritize_balanced:
    loss_weights = {'seg': 0.9, 'cdr_class': 1.1}
elif address_class_imbalance:
    clf_loss = weighted_categorical_crossentropy(class_weights)
    loss_weights = {'seg': 0.9, 'cdr_class': 1.1}
```

### Step 4: Train Model
```python
# Build model
model = build_att_unet3d_with_class(
    input_shape=(128,128,128,1),
    tabular_dim=7,
    base_filters=32,
    dropout_rate=0
)

# Compile
model.compile(
    optimizer=AdamW(learning_rate=2e-3, weight_decay=1e-5),
    loss={'seg': make_tversky_loss(0.5, 0.5), 'cdr_class': clf_loss},
    loss_weights=loss_weights,
    metrics={'seg': [dice_coef, jaccard_coef], 'cdr_class': ['accuracy']}
)

# Train
history = model.fit(
    train_ds.repeat(),
    epochs=20,
    steps_per_epoch=21,
    validation_data=val_ds,
    validation_steps=3
)
```

### Step 5: Evaluate
```python
# Test on validation set
val_results = test_and_save_multimodal_classification(
    model=model,
    test_pairs=val_pairs,
    df_features=val_df,
    feature_cols=feature_cols,
    target_col="CDR_Class",
    out_dir="val_results",
    num_classes=4
)
```

## Files Structure
```
Stage B/
├── README.md                              # This documentation
├── STAGE_B_final.ipynb                      # Complete implementation
├── val_results_AdamW_lw_0.8-1.2/     # AdamW, seg:0.8, cls:1.2 results
├── val_results_AdamW_lw_0.9-1.1/     # AdamW, seg:0.9, cls:1.1 results
└── val_results_AdamW_WCL_lw_0.9-1.1/ # AdamW, WCE, seg:0.9, cls:1.1 results
```

## Conclusion

Stage B demonstrates the critical importance of **loss weighting strategies** in multi-task learning. Key insights:

1. **Loss Weight Balance**: The ratio between segmentation and classification losses significantly impacts performance. Balanced weights (0.9/1.1) provided the best overall results.

2. **Class Imbalance Challenge**: While weighted categorical cross-entropy is theoretically sound, extreme class imbalances (1 sample for Very Mild) can cause prediction bias and training instability.

3. **Trade-offs**: Improving one task often degrades the other. Experiment 2 offered the best balance with 80% classification accuracy and reasonable segmentation performance (Dice: 0.6136).

4. **Dataset Limitations**: The small dataset (81 training samples) and severe class imbalance limit the effectiveness of advanced training strategies.

**Best Configuration**: Loss weights {seg: 0.9, cdr_class: 1.1} with standard categorical cross-entropy achieved the most stable and balanced performance.

---

**Note**: All experiments fully implemented. Results demonstrate the challenges of multi-task learning with imbalanced medical datasets and the need for careful hyperparameter tuning.