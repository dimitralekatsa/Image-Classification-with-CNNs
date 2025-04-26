# CNN Architecture Analysis and Implementation

## Overview
This project was created as part of the "Image and Video Technology Analysis" course but has been developed as a standalone exploration of convolutional neural networks (CNNs). The project focuses on analyzing leading CNN architectures and implementing a custom CNN model for image classification using the CIFAR-100 dataset.

## Project Structure

### 1. Theoretical Analysis
A comprehensive comparison of three groundbreaking CNN architectures:
- **LeNet**: The pioneer architecture for digit recognition
- **AlexNet**: The architecture that revolutionized computer vision in 2012
- **VGG**: A deeper architecture demonstrating the power of network depth

The analysis includes detailed comparison of:
- Network layers and architecture
- Filter sizes and receptive fields
- Activation functions
- Parameter counts and efficiency
- Pooling techniques
- Regularization methods including dropout

### 2. Practical Implementation
- Custom CNN implementation for image classification
- Working with the CIFAR-100 dataset
- Performance evaluation and optimization
- Visualization of results and model behavior

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow and Keras
- Jupyter Notebook

### Dependencies

```
# Core libraries
tensorflow>=2.0.0
keras
numpy
pandas
matplotlib
scikit-learn

# Additional libraries
tensorflow-datasets
h5py
```

### Installation

```
# Clone the repository
git clone https://github.com/yourusername/cnn-architecture-analysis.git
cd cnn-architecture-analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

1. Open the Jupyter notebook:
   ```
   jupyter notebook cnn_implementation.ipynb
   ```
2. Set your seed parameter for dataset consistency
3. Execute the notebook to reproduce the analysis and results

## Dataset

The project uses the CIFAR-100 dataset, which consists of 60,000 32x32 color images in 100 different classes. The dataset is automatically downloaded when running the notebook.

## Implementation Details

- Framework: TensorFlow/Keras
- Model Architecture: Custom CNN inspired by the analyzed architectures
- Training: Implemented with appropriate data augmentation and regularization techniques
- Evaluation: Includes accuracy metrics and confusion matrices
- Callbacks: Early stopping and model checkpointing for optimal training

## Results

The implementation achieves competitive accuracy on the CIFAR-100 test set while demonstrating the impact of architectural choices covered in the theoretical analysis.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original papers and implementations of LeNet, AlexNet, and VGG
- The CIFAR-100 dataset creators
