# Fashion MNIST Classification with PyTorch

A deep learning project implementing three different neural network architectures to classify Fashion MNIST dataset images, progressing from basic fully connected networks to convolutional neural networks (CNNs).

## 📋 Project Overview

This project explores image classification on the Fashion MNIST dataset using PyTorch, implementing and comparing three different model architectures:

1. **Model 01** - Basic Neural Network (CPU)
2. **Model 02** - Neural Network with ReLU activation (GPU)
3. **Model 03** - TinyVGG Convolutional Neural Network (GPU)

## 🎯 Results

| Model | Architecture | Accuracy | Training Time | Device |
|-------|-------------|----------|---------------|--------|
| Model 01 | Basic NN | ~85% | Baseline | CPU |
| Model 02 | NN + ReLU | ~87% | Faster | GPU |
| Model 03 | TinyVGG CNN | ~90%+ | Fastest | GPU |

## 🏗️ Model Architectures

### Model 01 - Baseline Neural Network
- Input Layer: 784 units (28x28 flattened images)
- Hidden Layer: 10 units
- Output Layer: 10 units (10 classes)
- Activation: None (Linear)

### Model 02 - Improved Neural Network
- Input Layer: 784 units
- Hidden Layer: 10 units with ReLU activation
- Output Layer: 10 units
- Optimizer: SGD (learning rate: 0.1)

### Model 03 - TinyVGG CNN
- **Conv Block 1**: 2x Conv2D layers (kernel=3, padding=1) + ReLU + MaxPool2D
- **Conv Block 2**: 2x Conv2D layers (kernel=3, padding=1) + ReLU + MaxPool2D
- **Classifier**: Flatten + Linear layer
- Hidden units: 10
- Significantly better performance on image data

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm torchmetrics mlxtend
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-mnist-pytorch.git
cd fashion-mnist-pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook or script:
```bash
python pytorch_computer_vision.py
```

## 📊 Dataset

**Fashion MNIST** consists of:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes of fashion items

### Classes
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## 🔬 Features

- **Data Visualization**: Sample images from the dataset with labels
- **Multiple Architectures**: Comparison between simple NN and CNN
- **GPU Acceleration**: Device-agnostic code for CPU/GPU training
- **Performance Metrics**: Accuracy, loss tracking, and confusion matrix
- **Model Persistence**: Save and load trained models
- **Inference Pipeline**: Make predictions on new images

## 📈 Training Process

The project implements:
- Cross-Entropy Loss function
- Stochastic Gradient Descent (SGD) optimizer
- Batch training with DataLoader (batch_size=32)
- Training/testing loop with progress tracking
- Model evaluation with accuracy metrics

## 🎨 Visualizations

The project includes:
- Random sample visualization from training data
- Batch visualization
- Prediction results with correct/incorrect labels (green/red)
- Confusion matrix for detailed performance analysis
- Model accuracy comparison bar chart

## 💾 Model Saving

The best performing model (TinyVGG) is saved as:
```
model/TinyVGG_Computer_vison_model.pth
```

Load the model:
```python
loaded_model = FashionMNISTv3(input_shape=1, hidden_units=10, output_shape=10)
loaded_model.load_state_dict(torch.load('model/TinyVGG_Computer_vison_model.pth'))
loaded_model.to(device)
```

## 🧪 Evaluation Metrics

- **Accuracy**: Primary metric for classification performance
- **Loss**: Cross-entropy loss for training/testing
- **Confusion Matrix**: Detailed per-class performance analysis
- **Training Time**: Comparison across different devices and architectures

## 📚 Learning Resources

This project was created by following Daniel Bourke's PyTorch course on YouTube. Special thanks to:
- [PyTorch for Deep Learning & Machine Learning](https://www.youtube.com/c/DanielBourke)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Daniel Bourke** for the excellent PyTorch tutorial series
- **Zalando Research** for the Fashion MNIST dataset
- **PyTorch Team** for the amazing deep learning framework

## 📧 Contact

Feel free to reach out if you have any questions or suggestions!

---

**Note**: This is a learning project focused on understanding PyTorch fundamentals and computer vision concepts.
