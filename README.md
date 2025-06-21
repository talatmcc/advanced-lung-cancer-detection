# Advanced Lung Cancer Detection using Knowledge Distillation

This repository contains a comprehensive deep-learning pipeline for lung cancer detection using knowledge distillation techniques. The project implements a multi-modal approach combining CT scans and histopathology images to achieve high-performance binary classification of lung cancer cases.

## 🎯 Project Overview

The project employs a teacher-student knowledge distillation framework where:
- **Teacher Models**: Pre-trained EfficientNet (B0) architectures serve as high-capacity teachers
- **Student Models**: Lightweight custom CNN3 architecture learn from teacher guidance
- **Multi-modal Input**: Processes both CT scans and histopathology images
- **Binary Classification**: Distinguishes between non-cancerous and cancerous cases

## 📁 Repository Structure

```
├── 01-lung-all-parent-seed42.ipynb                    # Teacher model training (EfficientNet-b0 as best parent/teacher model)
├── 02-lung-all-student-seed-42-cnn1-to-cnn5.ipynb   # Student model training CNN1-CNN5 (CNN3 performs best)
├── 03-kd-cnn3-temperature-2-alpha0-5-seed-42-gradcam++.ipynb  # Knowledge distillation (from EfficientNet-b0 to CNN3) + GradCAM++
├── 04-lung-tsne-visualization-cnn3.ipynb              # t-SNE feature visualization
├── 01-best_EfficientNetB0-seed42.pth                  # saved Pre-trained EfficientNet B0 model
├── 03-best_distilled_student_model-cnn3-seed42.pth    # saved Best distilled student model
└── README.md                                          
```

## 🔧 Environment Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM
- 50GB+ storage space

### Required Dependencies

```bash
# Core ML libraries
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
Pillow>=8.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Feature visualization
sklearn
umap-learn

# Model interpretation
grad-cam>=1.4.0

# Jupyter ecosystem
jupyter>=1.0.0
ipywidgets>=7.6.0
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd advance-lung-cancer-detection-codes

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # Create this file with above dependencies
```

## 📊 Dataset Requirements

### Dataset Download

The dataset is publicly available on Kaggle and can be downloaded from:

**🔗 [Lung Cancer Two-Class Dataset Combined-4](https://www.kaggle.com/datasets/talatmahmudtomal/lung-cancer-two-class-dataset-combined-4-tomal)**

### Download Instructions

1. **Via Kaggle Web Interface:**
   - Visit the dataset link above
   - Click "Download" button (requires Kaggle account)
   - Extract to your project directory

2. **Via Kaggle API:**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Configure API credentials (see Kaggle documentation)
   # Download dataset
   kaggle datasets download -d talatmahmudtomal/lung-cancer-two-class-dataset-combined-4-tomal
   
   # Extract dataset
   unzip lung-cancer-two-class-dataset-combined-4-tomal.zip
   ```



### Dataset Specifications
- **Image Format**: JPEG/PNG
- **Image Size**: 224×224 pixels (automatically resized)
- **Modalities**: CT scans and histopathology images
- **Classes**: Binary (non_cancerous, cancerous)
- **Class Distribution**: Imbalanced dataset with 11,600 cancerous and 5,877 non-cancerous images
- **Split Ratio**: 70% train, 15% validation, 15% test
- **Total Size**: ~4GB (when extracted)
- **Data Sources**: Four combined datasets (IQ-OTHNCCD, LC25000, Mendeley, Kaggle)

## 🚀 Usage Instructions

### 1. Parent Model Training (EfficientNet-b0)

Execute the parent model training notebook to establish teacher models:

```bash
jupyter notebook 01-lung-all-parent-seed42.ipynb
```

**Key Features:**
- Pre-trained EfficientNet-B0-, EfficientNet-B2, RESNET-50, VGG-16, DENSENET-121 architectures
- Transfer learning with ImageNet weights
- Comprehensive evaluation metrics
- Model checkpointing and early stopping

**Expected Outputs:**
- `01-best_EfficientNetB0-seed42.pth`: Best teacher model
- Training/validation curves
- Performance metrics and confusion matrices

### 2. Student Model Training

Train multiple student architectures to identify best student model:

```bash
jupyter notebook 02-lung-all-student-seed-42-cnn1-to-cnn5.ipynb
```

**Student Architectures:**
- **CNN1**: Ultra-minimal architecture (1,249 parameters, 0.01 MB)
- **CNN2**: Lightweight design (1,754 parameters, 0.01 MB)
- **CNN3**: Balanced architecture - **RECOMMENDED** (10,967 parameters, 0.05 MB)
- **CNN4**: Deeper network (105,344 parameters, 0.50 MB)
- **CNN5**: Complex architecture (1,210,402 parameters, 5.00 MB)

**CNN3 Architecture Details:**
- Input: 224×224×3 images
- Conv Layer 1: 3→2 channels, ReLU, MaxPool(2×2) → 112×112×2
- Conv Layer 2: 2→4 channels, ReLU, MaxPool(4×4) → 28×28×4  
- Conv Layer 3: 4→6 channels, ReLU, MaxPool(4×4) → 7×7×6
- Flatten → 1×1×294
- FC Layer 1: 294→9, ReLU
- FC Layer 2: 9→2, Softmax (binary classification)

**Training Configuration:**
- Batch size: 48
- Learning rate: 0.001 (Adam optimizer)
- Epochs: Up to 100 (with early stopping, patience=5)
- Scheduler: ReduceLROnPlateau (for teacher model)
- Weight decay: 1e-4 (for student model)

### 3. Knowledge Distillation with Visualization

Apply knowledge distillation and generate interpretability visualizations:

```bash
jupyter notebook 03-kd-cnn3-temperature-2-alpha0-5-seed-42-gradcam++.ipynb
```

**Distillation Parameters:**
- Temperature (T): 2.0
- Alpha (α): 0.5
- Loss: `α × CrossEntropy + (1-α) × KL_Divergence`

**Visualization Features:**
- GradCAM++ heatmaps for model interpretability
- Layer-wise activation analysis
- Attention mechanism visualization

### 4. Feature Space Analysis

Explore learned feature representations:

```bash
jupyter notebook 04-lung-tsne-visualization-cnn3.ipynb
```

**Analysis Components:**
- t-SNE dimensionality reduction
- Feature clustering by cancer type
- Modality separation analysis (CT vs. Histopathology)
- Class separability metrics

### t-SNE Analysis
- **Feature Extraction**: CNN feature maps from trained models
- **Dimensionality**: 2D projection for visualization
- **Insights**: 
  - Clear separation between cancer classes
  - Distinct clustering by imaging modality
  - Feature space organization patterns


## 🔍 Model Interpretability

### GradCAM++ Visualizations
- **Purpose**: Highlight image regions contributing to predictions
- **Implementation**: Layer-wise attention maps
- **Output**: Heatmap overlays on original images



## ⚙️ Reproducibility

### Experimental Setup
- **Hardware**: Kaggle platform with P100 GPU
- **CPU**: Intel(R) Core(TM) i5-8350U @ 1.70GHz, 16GB RAM


### Available Pre-trained Models
1. **EfficientNet B0 Teacher**: `01-best_EfficientNetB0-seed42.pth`
   - Architecture: EfficientNet-B0
   - Parameters: 4,010,110 (~4.01M)
   - Model Size: 15.58 MB


2. **CNN3 Student**: `03-best_distilled_student_model-cnn3-seed42.pth`
   - Architecture: Custom CNN3 (3 convolutional layers + 2 fully connected)
   - Parameters: 10,967 (ultra-lightweight)
   - Model Size: 0.05 MB


## 🚨 Important Notes

1. **Dataset Path Configuration**: After downloading the dataset, update the `data_dir` variable in each notebook:
   ```python
   # Change this path to match your local dataset location
   data_dir = "/path/to/your/lung-cancer-two-class-dataset-combined-4"
   ```
3. **GPU Memory**: Ensure sufficient GPU memory (8GB+ recommended) for batch processing
6. **Kaggle Account**: Required for dataset download via web interface

## 🤝 Contributing

When contributing to this project:
1. Maintain reproducibility by using seed=42
2. Follow existing code structure and documentation standards
3. Include performance benchmarks for new model variants
4. Update this README with any architectural changes

## 📄 License

This project is released under the MIT License. Please cite this work if used in academic research.




## 🏆 Key Achievements

- **State-of-the-art Efficiency**: 341× model compression with only 0.23% accuracy drop
- **Superior Speed**: 21× faster inference compared to teacher model
- **Clinical Relevance**: Interpretable predictions using Grad-CAM++ visualization
- **Cross-domain Generalization**: Validated on four diverse datasets
- **Resource Optimization**: Suitable for edge devices and real-time deployment
- **Comprehensive Evaluation**: Statistical significance across multiple random seeds

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size from 48 to 32 or 16
2. **Dataset Loading Errors**: Verify dataset structure matches expected format
3. **Model Loading Issues**: Ensure PyTorch versions match between training/inference
4. **Slow Training**: Enable `torch.backends.cudnn.benchmark = True` for fixed input sizes

### Support

For technical issues or questions:
1. Check existing notebook outputs for reference results
2. Verify environment setup matches requirements
3. Ensure dataset format compliance
4. Monitor GPU memory usage during training

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
**Tested Environment**: Python 3.8+ | PyTorch 1.12+ | CUDA 11.8+ 