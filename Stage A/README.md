# Stage A: Baseline 3D U-Net for Hippocampus Segmentation

## Overview

Stage A implements the baseline 3D U-Net architecture for automatic hippocampus segmentation from brain MRI scans. This stage serves as the foundation for comparing more advanced approaches in subsequent stages. It focuses on single-task learning using only MRI volume data.

## Architecture

### Model: 3D U-Net with Classification Head

The model combines a segmentation branch with a classification branch:

- **Encoder**: 3D convolutional layers with max pooling for feature extraction
- **Decoder**: 3D transposed convolutions with skip connections for segmentation
- **Classification Head**: Global average pooling + dense layers for CDR classification
- **Multi-output**: Simultaneous segmentation and classification predictions

### Key Components

```python
# Model Architecture Overview
inputs = Input(shape=(64, 64, 64, 1))

# Encoder path
conv1 = Conv3D(32, 3, activation='relu', padding='same')(inputs)
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# ... additional encoder layers ...

# Decoder path with skip connections
up6 = UpSampling3D(size=(2, 2, 2))(conv5)
merge6 = concatenate([conv4, up6], axis=4)

# ... additional decoder layers ...

# Segmentation output
seg_out = Conv3D(1, 1, activation='sigmoid', name='segmentation')(conv9)

# Classification output
gap = GlobalAveragePooling3D()(conv5)
dense1 = Dense(128, activation='relu')(gap)
class_out = Dense(4, activation='softmax', name='classification')(dense1)

model = Model(inputs=inputs, outputs=[seg_out, class_out])
```

## Data Processing

### Input Requirements

1. **MRI Volumes**: 3D T1-weighted brain scans
   - Format: `.nii` or `.mgz` files
   - Resolution: Resampled to 64×64×64 voxels
   - Preprocessing: Intensity normalization, skull stripping

2. **Ground Truth Labels**: Binary hippocampus masks
   - Format: `.nii` files
   - Values: 0 (background), 1 (hippocampus)

3. **Clinical Data**: Patient metadata for classification
   - CDR scores: Clinical Dementia Rating (0, 0.5, 1.0, 2.0)
   - Demographics: Age, gender, handedness
   - Cognitive scores: MMSE, brain volume measurements

### Data Preparation Pipeline

```python
# 1. Dataset loading and validation
train_csv, val_csv, test_csv = load_and_validate_datasets()

# 2. Preprocessing clinical data
train_df, encoders = preprocess_csv(train_csv, fit=True)
val_df, _ = preprocess_csv(val_csv, le_dict=encoders, fit=False)
test_df, _ = preprocess_csv(test_csv, le_dict=encoders, fit=False)

# 3. CDR classification encoding
cdr_map = {0.0: 'Healthy', 2.0: 'Very Mild', 1.0: 'Mild', 0.5: 'Moderate'}
fixed_label_order = ['Healthy', 'Very Mild', 'Mild', 'Moderate']
```

## Training Configuration

### Hyperparameters

- **Batch Size**: 4 (due to memory constraints with 3D volumes)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 50-100 with early stopping
- **Loss Functions**:
  - Segmentation: Binary cross-entropy
  - Classification: Categorical cross-entropy
- **Loss Weights**: Equal weighting (1.0 for both tasks)

### Training Process

```python
# Model compilation
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'segmentation': 'binary_crossentropy',
        'classification': 'categorical_crossentropy'
    },
    loss_weights={
        'segmentation': 1.0,
        'classification': 1.0
    },
    metrics={
        'segmentation': ['accuracy', dice_coefficient],
        'classification': ['accuracy']
    }
)

# Training with callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5),
    ModelCheckpoint('best_model_stage_a.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=callbacks
)
```

## Usage Instructions

### Step 1: Data Preparation

```bash
# Ensure you have the preprocessed dataset
# - Final_tvt_Dataset.zip (train/val/test splits)
# - strat_subset.csv (clinical metadata)
```

### Step 2: Run Stage A Training

```python
# Execute the notebook cells in order:
# 1. Import dependencies
# 2. Load and validate datasets
# 3. Preprocess clinical data
# 4. Build model architecture
# 5. Configure training parameters
# 6. Train the model
# 7. Evaluate performance
```

### Step 3: Model Evaluation

The notebook includes comprehensive evaluation:

- **Segmentation Metrics**: Dice coefficient, IoU, sensitivity, specificity
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Visualization**: Training curves, prediction examples

## Expected Results

### Performance Benchmarks

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Segmentation Dice | 0.85 | 0.82 | 0.81 |
| Segmentation IoU | 0.74 | 0.71 | 0.70 |
| Classification Accuracy | 0.78 | 0.75 | 0.73 |

### Training Characteristics

- **Convergence**: Typically converges within 30-50 epochs
- **Memory Usage**: ~8GB GPU memory for batch size 4
- **Training Time**: ~2-3 hours on GPU (V100/A100)

## Key Features

### Advantages

1. **Baseline Reference**: Provides benchmark for comparing advanced architectures
2. **Dual-task Learning**: Simultaneous segmentation and classification
3. **Standard Architecture**: Well-established 3D U-Net design
4. **Computational Efficiency**: Relatively fast training and inference

### Limitations

1. **Single Modality**: Uses only MRI volume data
2. **No Attention**: Lacks attention mechanisms for improved focus
3. **Simple Fusion**: Basic concatenation for multi-task learning
4. **Limited Context**: No incorporation of clinical tabular data

## Files Structure

```
Stage A/
├── README.md               # This documentation
└── STAGE_A_final.ipynb    # Complete implementation notebook
```

## Implementation Details

### Data Generators

Custom data generators handle:
- **Memory Management**: Efficient loading of large 3D volumes
- **Augmentation**: Real-time data augmentation during training
- **Batch Processing**: Proper batching for multi-task outputs

### Model Architecture Details

- **Input Shape**: (None, 64, 64, 64, 1)
- **Encoder Filters**: [32, 64, 128, 256, 512]
- **Decoder Filters**: [256, 128, 64, 32]
- **Classification Dense**: [128, 4] neurons
- **Activation Functions**: ReLU (hidden), Sigmoid (segmentation), Softmax (classification)

## Reproducibility

### Required Dependencies

```python
tensorflow>=2.8.0
nibabel>=3.2.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### Random Seed Configuration

```python
# Set seeds for reproducibility
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## Next Steps

After completing Stage A:

1. **Analyze Results**: Review performance metrics and identify areas for improvement
2. **Save Model**: Export trained weights for comparison with later stages
3. **Proceed to Stage B**: Implement attention mechanisms for enhanced performance
4. **Document Findings**: Record baseline metrics for research comparison

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or image resolution
2. **Convergence Issues**: Adjust learning rate or add regularization
3. **Data Loading**: Verify file paths and data format consistency
4. **Imbalanced Classes**: Consider class weighting for CDR classification

### Performance Optimization

- **Mixed Precision**: Enable for faster training on compatible GPUs
- **Data Pipeline**: Use `tf.data` for optimized data loading
- **Model Parallelism**: Distribute across multiple GPUs if available

---

**Note**: Stage A serves as the foundation for the entire research pipeline. Ensure thorough validation before proceeding to advanced stages.