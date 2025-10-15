# Automatic Hippocampus Segmentation and CDR Classification for Alzheimer's Disease

## Project Overview

This project implements a multimodal deep learning framework for automatic hippocampus segmentation from 3D T1-weighted brain MRI scans and simultaneous Clinical Dementia Rating (CDR) classification for Alzheimer's disease assessment. The pipeline combines 3D MRI volume data with clinical tabular metadata through a sophisticated joint learning paradigm.

## Our Contributions

1. **Dataset Selection and Preprocessing**
   - Developed a custom stratification strategy for OASIS-1 dataset
   - Created comprehensive preprocessing pipeline for 3D MRI data
   - Implemented data balancing techniques for robust model training

2. **Model Architecture and Training**
   - Built a multimodal framework combining MRI and clinical data
   - Implemented attention mechanisms in 3D U-Net architecture
   - Developed loss weighting strategies for multi-task learning

3. **Progressive Development (Stages A→D)**
   - Stage A: Base architecture with attention gates
   - Stage B: Loss weighting optimization
   - Stage C: Enhanced multimodal fusion
   - Stage D: Final joint learning framework with best performance

## Resources Used

1. **Base Architecture**
   - 3D U-Net structure: Adapted from public implementations
   - Attention mechanisms: Based on standard attention modules
   - Loss functions: Standard implementations (Dice loss, categorical cross-entropy)

2. **External Libraries**
   - TensorFlow/Keras for model implementation
   - NiBabel for MRI data handling
   - SimpleITK for image preprocessing
   - Standard Python libraries (NumPy, Pandas, scikit-learn)

3. **Dataset**
   - OASIS-1 dataset: Publicly available from Washington University School of Medicine
   - Original data format and documentation from OASIS project

## How to Use This Code

### Prerequisites
1. Install required packages:
```bash
pip install tensorflow nibabel SimpleITK numpy pandas scikit-learn
```

### Step-by-Step Guide

1. **Data Preprocessing**
   - Run `3D_Preprocessing.ipynb` to prepare the MRI data
   - Run `cs760_subset_selection.ipynb` to create the stratified dataset

2. **Model Training and Evaluation**
   - For best results, run the notebooks in this order:
     1. `Stage A/STAGE_A_final.ipynb`: Base implementation
     2. `Stage B/STAGE_B_final.ipynb`: Loss weighting
     3. `Stage C/STAGE_C_final.ipynb`: Multimodal fusion
     4. `Stage D/STAGE_D_V2_final.ipynb`: Final recommended version

### Replicating Our Best Results

To replicate our best performance (90.91% classification accuracy, 0.7943 Dice score):

1. Use `STAGE_D_V2_final.ipynb` in the Stage D folder
2. Ensure you've run the preprocessing notebooks first
3. Use the following configuration:
   - AdamW optimizer with learning rate 3e-4
   - Weight decay: 1e-4
   - Dropout rate: 0.2
   - Batch size: As specified in the notebook
   - Training epochs: Follow the notebook parameters

## Project Structure and Results

The project is organized into four progressive stages (A→B→C→D), each building upon the previous stage's achievements:

### Stage A: Baseline Multimodal 3D Attention U-Net
- Implemented base architecture with attention gates and multimodal fusion
- Combined 3D MRI volumes with clinical tabular data
- Established foundation for joint segmentation and CDR classification

### Stage B: Loss-Weighted Framework
- Introduced loss weighting strategies for task balancing
- Implemented weighted categorical cross-entropy for class imbalance
- Explored multiple loss weight configurations
- Maintained architecture from Stage A while optimizing training strategy

### Stage C: Enhanced Multimodal Fusion
- Developed sophisticated fusion of MRI and clinical data
- Integrated 7 key clinical features (demographics, cognitive scores, brain measurements)
- Implemented late fusion at the bottleneck via concatenation
- Added multi-task outputs for both segmentation and CDR classification

### Stage D: Final Joint Learning Framework (Best Performance)
Three variants were implemented with the following results:

#### STAGE_D_V2_final (Recommended Version)
- Training/Validation/Test Split: 81/10/11 subjects
- Segmentation Performance:
  - Mean Dice: 0.7943
  - Mean Jaccard: 0.6899
- Classification Accuracy: 90.91%

#### STAGE_D_DV_final
- Training/Validation/Test Split: 81/11/11 subjects
- Segmentation Performance:
  - Mean Dice: 0.8202
  - Mean Jaccard: 0.7019
- Classification Accuracy: 90.91%

#### STAGE_D_DV1_final
- Training/Validation/Test Split: 81/10/11 subjects
- Segmentation Performance:
  - Mean Dice: 0.8678
  - Mean Jaccard: 0.7689

## Technical Implementation

### Architecture Components
1. **3D Attention U-Net Backbone**
   - 4-level encoder with base filters ranging from 32 to 256 channels
   - Attention gates in decoder skip connections
   - Bottleneck with 512 channels

2. **Multimodal Fusion Module**
   - MRI Branch: 3D convolutional encoder-decoder
   - Clinical Data Branch: Dense network for 7 tabular features
   - Late fusion at bottleneck via concatenation

3. **Dual Task Outputs**
   - Segmentation: Binary mask with sigmoid activation
   - Classification: 4-class CDR prediction (Healthy, Very Mild, Mild, Moderate)

### Key Features
- **Attention Mechanisms**: Enhanced feature localization
- **Loss Weighting**: Dynamic balancing between tasks
- **Clinical Features**: Integration of key demographic and cognitive metrics
- **Dropout Regularization**: Configurable dropout rate (0.2 default)

### Training Configuration
- Optimizer: AdamW with learning rate 1e-3 to 3e-4
- Weight Decay: 1e-4
- Loss Functions: Combination of Dice loss and weighted categorical cross-entropy
- Data Augmentation: 3D-specific augmentation techniques

## Dataset and Preprocessing
- Based on OASIS-1 dataset from Washington University School of Medicine
- Custom stratification strategy for balanced representation
- Preprocessed 3D T1-weighted MRI scans
- Integrated clinical metadata including:
  - Demographics
  - MMSE scores
  - CDR ratings
  - Brain volume measurements

## Project Structure
```
760_Automatic_Hippocampus_Segmentation/
├── README.md                          # Main project documentation
├── 3D_Preprocessing.ipynb            # Data preprocessing pipeline
├── cs760_subset_selection.ipynb     # Dataset stratification and subset selection
├── Stage A/                         # Baseline implementation
│   ├── README.md                    # Stage A documentation
│   └── STAGE_A_final.ipynb         # Implementation and results
├── Stage B/                         # Loss weighting implementation
│   ├── README.md                    # Stage B documentation
│   └── STAGE_B_final.ipynb         # Implementation and results
├── Stage C/                         # Multimodal fusion implementation
│   ├── README.md                    # Stage C documentation
│   └── STAGE_C_final.ipynb         # Implementation and results
└── Stage D/                         # Final joint learning implementation
    ├── README.md                    # Stage D documentation
    ├── STAGE_D_V2_final.ipynb      # Recommended version
    ├── STAGE_D_DV_final.ipynb      # Alternative version
    └── STAGE_D_DV1_final.ipynb     # Version 1 implementation
```

## Dependencies
- TensorFlow/Keras for deep learning
- NiBabel for neuroimaging data handling
- SimpleITK for medical image preprocessing
- NumPy/Pandas for data manipulation
- scikit-learn for evaluation metrics

## Conclusion
The final model (STAGE_D_V2_final) achieves robust performance in both hippocampus segmentation (Dice score: 0.7943) and CDR classification (accuracy: 90.91%), demonstrating the effectiveness of our joint learning approach for Alzheimer's disease assessment.
3. **Attention-Enhanced 3D U-Net** with cross-task interaction mechanisms
4. **Adaptive Training Methodology** that balances multiple objectives dynamically
5. **Clinical Translation Focus** with uncertainty quantification and explainable AI components

