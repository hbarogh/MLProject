# AI Image Detection Project

## Project Overview
This project builds and trains a custom **Convolutional Neural Network (CNN)** to classify images as either **real** (from ImageNet100) or **AI-generated** (e.g., from DALL-E-2).  
The model detects synthetic images based on subtle visual patterns.

It uses **PyTorch Lightning** for modular training, applies **balanced sampling** to handle class imbalance, and includes **data augmentation** to boost generalization.

---

## Technologies and Libraries Used
- **Python 3.10+**
- **PyTorch** - Core deep learning framework
- **PyTorch Lightning** - Simplifies training and logging
- **TorchMetrics** - For easy accuracy tracking
- **Torchvision** - Image transformations and utilities
- **Pillow (PIL)** - Image loading
- **NumPy** (optional but helpful)

---

## How It Works

### 1. Dataset Structure
- **Real images**: Pulled from ImageNet100 (four train splits + validation set)
- **Fake images**: AI-generated `.png` images (e.g., DALL-E-2)

Images are organized into three splits:
- **Train**: Model training with augmentations
- **Validation**: For model selection
- **Test**: Final model evaluation

The real/fake datasets are split carefully so there’s no overlap between train, validation, and test.

---

### 2. Data Loading
- Custom `CombinedAIDataset` reads both real and fake images and assigns binary labels.
- **Training set** uses a **WeightedRandomSampler** to rebalance the classes dynamically every epoch.
- **Transformations**:
  - **Training images**:
    - Resize to 256×256
    - Random horizontal flip
    - Color jitter (brightness/contrast)
    - Random rotation (±15 degrees)
    - Normalize using ImageNet statistics
  - **Validation and test images**:
    - Resize to 256×256
    - Normalize (no augmentations)

---

### 3. Model Architecture
- **Convolutional Layers**:
  - 4 convolutional blocks (32 → 64 → 128 → 256 filters)
  - Each block uses Batch Normalization and ReLU activations
  - MaxPooling layers reduce spatial size
- **Global Feature Extraction**:
  - Adaptive Average Pooling to (4x4) feature maps
- **Fully Connected Layers**:
  - Linear → ReLU → Dropout(0.3) → Linear to 1 output
- **Binary Classification**:
  - Output is a single logit (passed through sigmoid during evaluation)

---

### 4. Training Details
- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), using a **positive class weight** to rebalance learning
- **Optimizer**: Adam (learning rate = 0.0001, weight decay = 0.0001)
- **Learning Rate Scheduler**: `ReduceLROnPlateau` lowers LR if validation loss plateaus
- **Metrics**: Binary Accuracy
- **Mixed Precision**: 16-bit Automatic Mixed Precision (AMP) for faster training
- **GPU**: CUDA-enabled GPU used automatically if available

**Training Callbacks**:
- **EarlyStopping**: Stops training if validation loss doesn't improve for 5 epochs
- **ModelCheckpoint**: Saves the model with the lowest validation loss

---

## How to Run

1. Install dependencies:
    ```bash
    pip install torch torchvision lightning torchmetrics pillow
    ```

2. Organize your datasets:
    ```
    datasets/
        ImageNet100/
            train.X1/
            train.X2/
            train.X3/
            train.X4/
            val.X/
        ai/
            (all AI-generated .png images)
    ```

3. Run the training script:
    ```bash
    python your_script_name.py
    ```

---

## Key Features
- Custom CNN model (built from scratch)
- Clean PyTorch Lightning training loop
- Balanced training with WeightedRandomSampler
- Strong data augmentation for generalization
- Mixed precision and GPU acceleration
- Automatic model checkpointing and early stopping
- Learning rate scheduling for dynamic optimization

---

## Potential Future Improvements
- Train on larger real image pools (don't downsample real images)
- Switch to a deeper model (e.g., ResNet-based) with transfer learning
- Visualize model attention using GradCAM
- Fine-tune threshold selection beyond basic sigmoid(0.5) decision

---

## Credits
Developed as part of a deep learning project to explore classification of AI-generated vs. real-world images.  
Focus: building, optimizing, and improving a fully custom CNN.

---

## License
This project is open-source and available for educational purposes.
